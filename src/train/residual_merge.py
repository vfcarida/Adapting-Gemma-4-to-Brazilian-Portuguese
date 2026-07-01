"""Residual merge via advanced merging techniques for instruction recovery.

This module implements:
1. Task Arithmetic (Ilharco et al., 2023)
2. TIES-Merging (Yadav et al., 2023)
3. DARE-Linear (Yu et al., 2023)
4. DARE-TIES (Yu et al., 2023)

These advanced merging techniques resolve weight interference and parameter redundancy
when transferring instruction-following capability to a continued pretrained model.

Memory Management:
- We load base and instruct models sequentially to get their state dicts.
- We then load the CPT model once, merge the weights in-place directly on its parameters,
  and save the modified model. This avoids reloading the CPT model and keeps RAM usage minimal.
"""

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _trim_tensor(v: torch.Tensor, density: float) -> torch.Tensor:
    """Trim a tensor, keeping only the top-k magnitude values."""
    if density >= 1.0:
        return v

    flat = v.reshape(-1)
    k = int(density * flat.numel())
    if k <= 0:
        return torch.zeros_like(v)
    if k >= flat.numel():
        return v

    abs_flat = torch.abs(flat)
    threshold, _ = torch.kthvalue(abs_flat, flat.numel() - k + 1)

    mask = abs_flat >= threshold
    return v * mask.reshape(v.shape)


def _ties_elect_and_merge(trimmed_vectors: list[torch.Tensor]) -> torch.Tensor:
    """Elect consensus sign and perform disjoint merge on task vectors."""
    stacked = torch.stack(trimmed_vectors)  # (num_models, ...)

    # Sign of updates
    signs = torch.sign(stacked)

    # Magnitude sums for positive and negative signs
    pos_mask = stacked > 0
    neg_mask = stacked < 0

    pos_mag = torch.sum(torch.abs(stacked) * pos_mask.float(), dim=0)
    neg_mag = torch.sum(torch.abs(stacked) * neg_mask.float(), dim=0)

    # Consensus sign: 1 if positive magnitude is greater, else -1
    consensus_sign = torch.where(pos_mag >= neg_mag, 1.0, -1.0)

    # Keep only updates that agree with the consensus sign
    agreement_mask = (signs == consensus_sign) & (stacked != 0)

    # Sum of agreeing updates
    agreeing_sum = torch.sum(stacked * agreement_mask.float(), dim=0)

    # Count of agreeing updates
    agreeing_count = torch.sum(agreement_mask.float(), dim=0)

    # Average agreeing updates (avoid division by zero)
    merged = torch.where(agreeing_count > 0, agreeing_sum / agreeing_count, 0.0)

    return merged


def merge_tensors(
    base_w: torch.Tensor,
    cpt_w: torch.Tensor,
    instruct_w: torch.Tensor,
    alpha: float,
    method: str = "task_arithmetic",
    density: float = 1.0,
    seed: int = 42,
) -> torch.Tensor:
    """Merge base, CPT, and instruct tensors using the specified method."""
    orig_dtype = base_w.dtype
    device = base_w.device

    # Fast path for standard task arithmetic
    if method == "task_arithmetic" and density == 1.0:
        return (cpt_w.to(torch.float32) + alpha * (instruct_w.to(torch.float32) - base_w.to(torch.float32))).to(orig_dtype)

    base_f32 = base_w.to(torch.float32)
    cpt_f32 = cpt_w.to(torch.float32)
    instruct_f32 = instruct_w.to(torch.float32)

    v1 = cpt_f32 - base_f32
    v2 = instruct_f32 - base_f32

    if method == "task_arithmetic":
        v_merged = v1 + alpha * v2
        return (base_f32 + v_merged).to(orig_dtype)

    elif method == "dare_linear":
        if density <= 0.0 or density > 1.0:
            raise ValueError(f"Density must be in (0, 1], got {density}")

        generator = torch.Generator(device=device).manual_seed(seed)

        mask1 = (torch.rand(v1.shape, device=device, generator=generator) < density).float()
        mask2 = (torch.rand(v2.shape, device=device, generator=generator) < density).float()

        v1_dare = (v1 / density) * mask1
        v2_dare = (v2 / density) * mask2

        v_merged = v1_dare + alpha * v2_dare
        return (base_f32 + v_merged).to(orig_dtype)

    elif method == "ties":
        if density <= 0.0 or density > 1.0:
            raise ValueError(f"Density must be in (0, 1], got {density}")

        v2_scaled = alpha * v2
        v1_trimmed = _trim_tensor(v1, density)
        v2_trimmed = _trim_tensor(v2_scaled, density)

        v_merged = _ties_elect_and_merge([v1_trimmed, v2_trimmed])
        return (base_f32 + v_merged).to(orig_dtype)

    elif method == "dare_ties":
        if density <= 0.0 or density > 1.0:
            raise ValueError(f"Density must be in (0, 1], got {density}")

        generator = torch.Generator(device=device).manual_seed(seed)

        mask1 = (torch.rand(v1.shape, device=device, generator=generator) < density).float()
        mask2 = (torch.rand(v2.shape, device=device, generator=generator) < density).float()

        v1_dare = (v1 / density) * mask1
        v2_dare = ((alpha * v2) / density) * mask2

        v_merged = _ties_elect_and_merge([v1_dare, v2_dare])
        return (base_f32 + v_merged).to(orig_dtype)

    else:
        raise ValueError(f"Unknown merging method: {method}")


def compute_residual_merge(
    base_model_id: str,
    instruct_model_id: str,
    cpt_model_path: str,
    alpha: float,
    output_dir: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    method: str = "task_arithmetic",
    density: float = 1.0,
    seed: int = 42,
) -> Path:
    """Compute residual merge using in-place operations to minimize RAM usage."""
    output_path = Path(output_dir) / f"alpha_{alpha:.2f}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing residual merge using method={method} (alpha={alpha:.2f}, density={density})")
    logger.info(f"  Base: {base_model_id}")
    logger.info(f"  Instruct: {instruct_model_id}")
    logger.info(f"  CPT: {cpt_model_path}")

    start_time = time.time()

    # --- Load state dicts sequentially to minimize RAM ---

    logger.info("Loading base model weights...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=dtype, device_map=device
    )
    base_state = base_model.state_dict()
    del base_model
    gc.collect()

    logger.info("Loading instruct model weights...")
    instruct_model = AutoModelForCausalLM.from_pretrained(
        instruct_model_id, torch_dtype=dtype, device_map=device
    )
    instruct_state = instruct_model.state_dict()
    del instruct_model
    gc.collect()

    logger.info("Loading CPT model for in-place modification...")
    cpt_model = AutoModelForCausalLM.from_pretrained(
        cpt_model_path, torch_dtype=dtype, device_map=device
    )

    # --- Validate parameter compatibility ---

    base_keys = set(base_state.keys())
    instruct_keys = set(instruct_state.keys())
    cpt_state = cpt_model.state_dict()
    cpt_keys = set(cpt_state.keys())

    common_keys = base_keys & instruct_keys & cpt_keys
    logger.info(f"Common keys: {len(common_keys)} / {len(cpt_keys)} total")

    # --- Compute merge in-place directly on CPT model parameters ---

    logger.info(f"Merging weights in-place...")
    shape_mismatches = []

    with torch.no_grad():
        for key in tqdm(common_keys, desc="Merging"):
            base_w = base_state[key]
            instruct_w = instruct_state[key]
            cpt_w = cpt_state[key]

            # Validate shape
            if base_w.shape != instruct_w.shape or base_w.shape != cpt_w.shape:
                shape_mismatches.append(key)
                continue

            # Only perform merge arithmetic on floating point weights
            if not torch.is_floating_point(cpt_w):
                continue

            merged_w = merge_tensors(
                base_w=base_w,
                cpt_w=cpt_w,
                instruct_w=instruct_w,
                alpha=alpha,
                method=method,
                density=density,
                seed=seed,
            )

            # Copy in-place
            cpt_w.copy_(merged_w)

    if shape_mismatches:
        logger.warning(
            f"Shape mismatches (fell back to CPT weights): "
            f"{len(shape_mismatches)} params: {shape_mismatches[:5]}"
        )

    # Free base and instruct state dicts to release RAM before saving
    del base_state, instruct_state
    gc.collect()

    # --- Save merged model ---

    logger.info(f"Saving merged model to {output_path}")
    cpt_model.save_pretrained(output_path)
    del cpt_model
    gc.collect()

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cpt_model_path)
    tokenizer.save_pretrained(output_path)

    # Save metadata
    elapsed = time.time() - start_time
    metadata = {
        "method": method,
        "alpha": alpha,
        "density": density,
        "seed": seed,
        "base_model_id": base_model_id,
        "instruct_model_id": instruct_model_id,
        "cpt_model_path": str(cpt_model_path),
        "num_params_merged": len(common_keys),
        "num_shape_mismatches": len(shape_mismatches),
        "shape_mismatch_keys": shape_mismatches[:20],
        "elapsed_seconds": elapsed,
        "dtype": str(dtype),
    }
    with open(output_path / "merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Merge complete in {elapsed:.1f}s (method={method}, alpha={alpha})")
    return output_path


def alpha_sweep(
    base_model_id: str,
    instruct_model_id: str,
    cpt_model_path: str,
    alphas: list[float],
    output_dir: str,
    device: str = "cpu",
    method: str = "task_arithmetic",
    density: float = 1.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run merge for multiple alpha values to find optimal scaling."""
    output_dir = Path(output_dir)
    results = []

    for alpha in alphas:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Alpha sweep: {alpha:.2f}")
        logger.info(f"{'=' * 60}")

        path = compute_residual_merge(
            base_model_id=base_model_id,
            instruct_model_id=instruct_model_id,
            cpt_model_path=cpt_model_path,
            alpha=alpha,
            output_dir=str(output_dir),
            device=device,
            method=method,
            density=density,
            seed=seed,
        )
        results.append({"alpha": alpha, "path": str(path)})

    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    """CLI entry point for residual merge."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Advanced Weight Merging (TIES, DARE, Task Arithmetic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-model", type=str, required=True, help="Base model HF ID")
    parser.add_argument("--instruct-model", type=str, required=True, help="Instruct model HF ID")
    parser.add_argument("--cpt-model", type=str, required=True, help="Path to CPT model")
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        help="Alpha value(s) for instruction residual scaling",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/residual_merge")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--method",
        type=str,
        default="task_arithmetic",
        choices=["task_arithmetic", "ties", "dare_linear", "dare_ties"],
        help="Weight merging algorithm to use",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Density / keep probability fraction for TIES/DARE",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for DARE drop reproducibility",
    )

    args = parser.parse_args()

    if len(args.alpha) == 1:
        compute_residual_merge(
            base_model_id=args.base_model,
            instruct_model_id=args.instruct_model,
            cpt_model_path=args.cpt_model,
            alpha=args.alpha[0],
            output_dir=args.output_dir,
            device=args.device,
            method=args.method,
            density=args.density,
            seed=args.seed,
        )
    else:
        alpha_sweep(
            base_model_id=args.base_model,
            instruct_model_id=args.instruct_model,
            cpt_model_path=args.cpt_model,
            alphas=args.alpha,
            output_dir=args.output_dir,
            device=args.device,
            method=args.method,
            density=args.density,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
