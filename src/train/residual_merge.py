"""Residual merge via task arithmetic for instruction recovery.

This module implements the "task arithmetic" method from Ilharco et al. (2023)
to recover instruction-following capability after continued pretraining.

The key insight: instruction-tuning creates a "task vector" (the difference
between IT and base weights). This vector can be added to any checkpoint
derived from the same base model to transfer instruction capability.

Formula:
    instruction_residual = instruct_weights - base_weights
    adapted_instruct = cpt_weights + alpha * instruction_residual

Where:
    - base_weights: Original pre-trained model (e.g., gemma-4-E4B)
    - instruct_weights: Official instruction-tuned model (e.g., gemma-4-E4B-it)
    - cpt_weights: Our CPT-adapted model (trained from base on Aurora-PT)
    - alpha: Scaling factor controlling instruction strength

Alpha interpretation:
    - alpha = 0: Pure CPT model (no instruction capability)
    - alpha = 1: Full instruction vector transfer
    - alpha > 1: Amplified instructions (may degrade quality)
    - alpha < 1: Partial transfer (may preserve more CPT adaptation)

Memory management:
    Loading 3 full models simultaneously requires significant RAM.
    We load/unload models sequentially, keeping only state_dicts in memory.
    For a 4B model in bfloat16, each state_dict is ~8GB, so peak usage is ~24GB RAM.

References:
    - Ilharco et al. "Editing Models with Task Arithmetic" (ICLR 2023)
    - Yadav et al. "TIES-Merging" (NeurIPS 2023)
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


def _load_state_dict(
    model_path: str,
    dtype: torch.dtype,
    device: str,
    base_model_id: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load a model's state dict, transparently merging a LoRA adapter first.

    `cpt_model_path` may point at either a full model checkpoint (CPT ran
    with `use_lora: false`) or a LoRA adapter directory (CPT ran with
    `use_lora: true`, the default in configs/train/cpt_pilot.yaml — the
    original version of this function assumed a full checkpoint and would
    raise trying to `AutoModelForCausalLM.from_pretrained` an adapter-only
    directory). Adapter directories are detected via `adapter_config.json`
    and merged onto `base_model_id` via `merge_and_unload()` before the
    residual-merge arithmetic runs.

    Args:
        model_path: Path (or Hub ID) to load.
        dtype: Storage dtype for the loaded model.
        device: Device map for loading.
        base_model_id: Base model to merge a LoRA adapter onto. Required if
            `model_path` is a LoRA adapter directory.

    Returns:
        The (possibly LoRA-merged) model's state dict.
    """
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        if base_model_id is None:
            raise ValueError(
                f"{model_path} is a LoRA adapter directory but no base_model_id "
                "was provided to merge it onto."
            )
        logger.info(f"{model_path} is a LoRA adapter — merging onto {base_model_id} first")
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=dtype, device_map=device)
        merged = PeftModel.from_pretrained(base, model_path)
        merged = merged.merge_and_unload()
        state = merged.state_dict()
        del merged, base
        gc.collect()
        return state

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=device)
    state = model.state_dict()
    del model
    gc.collect()
    return state


def compute_residual_merge(
    base_model_id: str,
    instruct_model_id: str,
    cpt_model_path: str,
    alpha: float,
    output_dir: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    compute_dtype: torch.dtype = torch.float32,
) -> Path:
    """Compute residual merge and save the result.

    Loads all three models (base, instruct, CPT), computes the instruction
    residual, applies it to the CPT weights with scaling factor alpha,
    and saves the merged model.

    IMPORTANT: Arithmetic is performed in float32 (compute_dtype) to avoid
    accumulation errors from bfloat16. The final merged weights are cast
    back to the storage dtype (bfloat16) for saving.

    Args:
        base_model_id: HuggingFace ID of the original base model.
            Must be the exact model that was continued-pretrained.
        instruct_model_id: HuggingFace ID of the instruction-tuned variant.
            Must share the same architecture as base_model_id.
        cpt_model_path: Local path to the CPT-adapted model checkpoint.
        alpha: Scaling factor for the instruction residual.
            Typical range: [0.5, 1.2]. Start with 1.0.
        output_dir: Directory to save merged model. A subdirectory
            `alpha_X.XX` will be created for each alpha value.
        device: Device for computation. Use "cpu" to avoid GPU memory issues.
        dtype: Data type for model loading and final storage.
        compute_dtype: Data type for merge arithmetic (float32 recommended
            to avoid precision loss in subtraction/addition).

    Returns:
        Path to the saved merged model directory.

    Raises:
        RuntimeError: If model loading fails or shapes are incompatible.
    """
    output_path = Path(output_dir) / f"alpha_{alpha:.2f}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing residual merge with alpha={alpha}")
    logger.info(f"  Base: {base_model_id}")
    logger.info(f"  Instruct: {instruct_model_id}")
    logger.info(f"  CPT: {cpt_model_path}")

    start_time = time.time()

    # --- Load state dicts sequentially to minimize peak memory ---

    logger.info("Loading base model weights...")
    base_state = _load_state_dict(base_model_id, dtype, device)

    logger.info("Loading instruct model weights...")
    instruct_state = _load_state_dict(instruct_model_id, dtype, device)

    logger.info("Loading CPT model weights...")
    # Transparently merges a LoRA adapter onto base_model_id first if
    # cpt_model_path is an adapter directory (see _load_state_dict).
    cpt_state = _load_state_dict(cpt_model_path, dtype, device, base_model_id=base_model_id)

    # --- Validate parameter compatibility ---

    logger.info("Validating parameter shapes...")
    base_keys = set(base_state.keys())
    instruct_keys = set(instruct_state.keys())
    cpt_keys = set(cpt_state.keys())

    # Warn about key mismatches (shouldn't happen with same model family)
    if base_keys != instruct_keys:
        missing = base_keys - instruct_keys
        extra = instruct_keys - base_keys
        if missing:
            logger.warning(f"Keys in base but not instruct: {list(missing)[:5]}")
        if extra:
            logger.warning(f"Keys in instruct but not base: {list(extra)[:5]}")

    # Only merge parameters present in all three models
    common_keys = base_keys & instruct_keys & cpt_keys
    logger.info(f"Common keys: {len(common_keys)} / {len(cpt_keys)} total")

    # --- Compute merge: cpt + alpha * (instruct - base) ---

    logger.info(f"Computing merge (alpha={alpha}, compute_dtype={compute_dtype})...")
    merged_state = {}
    shape_mismatches = []

    # Pop each tensor out of its source state dict as it's consumed, instead
    # of keeping all three source dicts fully alive for the whole loop. This
    # keeps peak memory close to ~3x model size (three sources shrinking as
    # `merged_state` grows) rather than ~4x (three sources held in full plus
    # a fully-grown `merged_state` at the end of the loop).
    for key in tqdm(list(common_keys), desc="Merging"):
        base_w = base_state.pop(key)
        instruct_w = instruct_state.pop(key)
        cpt_w = cpt_state.pop(key)

        # Shape validation - all three must match for arithmetic
        if base_w.shape != instruct_w.shape or base_w.shape != cpt_w.shape:
            shape_mismatches.append(key)
            # Fallback: use CPT weights unchanged (safe default)
            merged_state[key] = cpt_w
            continue

        # Core task arithmetic in float32 to avoid precision loss
        # (bfloat16 subtraction can accumulate significant errors)
        residual = instruct_w.to(compute_dtype) - base_w.to(compute_dtype)
        merged = cpt_w.to(compute_dtype) + alpha * residual
        # Cast back to storage dtype for saving
        merged_state[key] = merged.to(dtype)

    if shape_mismatches:
        logger.warning(
            f"Shape mismatches (fell back to CPT weights): "
            f"{len(shape_mismatches)} params: {shape_mismatches[:5]}"
        )

    # Include CPT-only keys (e.g., added during LoRA merge)
    for key in cpt_keys - common_keys:
        merged_state[key] = cpt_state.pop(key)

    # Free (now-empty) source state dicts
    del base_state, instruct_state, cpt_state
    gc.collect()

    # --- Save merged model ---

    logger.info(f"Saving merged model to {output_path}")

    # Load model architecture from the BASE model (not cpt_model_path): if
    # CPT ran with LoRA, cpt_model_path is an adapter directory with no
    # standalone config.json and can't be loaded via from_pretrained here.
    # base_model_id is guaranteed to be a full model repo with the same
    # architecture (required for the arithmetic above to be valid at all).
    save_model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=dtype, device_map=device)
    save_model.load_state_dict(merged_state, strict=False)
    del merged_state  # now copied into save_model's parameters; free the dict
    gc.collect()
    save_model.save_pretrained(output_path)
    del save_model
    gc.collect()

    # Save the INSTRUCT model's tokenizer, not the CPT checkpoint's or
    # base's. The merged model is semantically instruction-following (that's
    # the entire point of adding the instruction residual), so it needs the
    # tokenizer that actually carries a working `chat_template` — verified
    # live: a Gemma 4 BASE tokenizer (google/gemma-4-E2B) has
    # `chat_template=None`, only the `-it` tokenizer has one. CPT doesn't
    # change vocab/add tokens in this pipeline (no vocab_expansion path is
    # wired up — see configs/model/*.yaml's `vocab_expansion.enabled:
    # false`), so base/instruct/CPT all share the same vocabulary and using
    # the instruct tokenizer loses nothing.
    tokenizer = AutoTokenizer.from_pretrained(instruct_model_id)
    tokenizer.save_pretrained(output_path)

    # Save metadata for reproducibility
    elapsed = time.time() - start_time
    metadata = {
        "method": "residual_merge_task_arithmetic",
        "formula": "cpt_weights + alpha * (instruct_weights - base_weights)",
        "base_model_id": base_model_id,
        "instruct_model_id": instruct_model_id,
        "cpt_model_path": str(cpt_model_path),
        "alpha": alpha,
        "num_params_merged": len(common_keys),
        "num_shape_mismatches": len(shape_mismatches),
        "shape_mismatch_keys": shape_mismatches[:20],
        "elapsed_seconds": elapsed,
        "storage_dtype": str(dtype),
        "compute_dtype": str(compute_dtype),
    }
    with open(output_path / "merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Merge complete in {elapsed:.1f}s (alpha={alpha})")
    return output_path


def alpha_sweep(
    base_model_id: str,
    instruct_model_id: str,
    cpt_model_path: str,
    alphas: list[float],
    output_dir: str,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run merge for multiple alpha values to find optimal scaling.

    Each alpha produces a separate model. After sweep, evaluate all models
    on benchmarks to find the best alpha (typically between 0.7 and 1.0).

    Args:
        base_model_id: Original base model ID.
        instruct_model_id: Original instruct model ID.
        cpt_model_path: Path to CPT checkpoint.
        alphas: List of alpha values to try.
        output_dir: Parent directory for all merged models.
        device: Computation device.

    Returns:
        List of {"alpha": float, "path": str} for each completed merge.
    """
    output_dir = Path(output_dir)
    results = []

    for alpha in alphas:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Alpha sweep: {alpha}")
        logger.info(f"{'=' * 60}")

        path = compute_residual_merge(
            base_model_id=base_model_id,
            instruct_model_id=instruct_model_id,
            cpt_model_path=cpt_model_path,
            alpha=alpha,
            output_dir=str(output_dir),
            device=device,
        )
        results.append({"alpha": alpha, "path": str(path)})

    # Save sweep summary for easy lookup
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    """CLI entry point for residual merge."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Residual Merge (Task Arithmetic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single alpha
  python -m src.train.residual_merge \\
      --base-model google/gemma-4-E4B \\
      --instruct-model google/gemma-4-E4B-it \\
      --cpt-model outputs/cpt_main/final \\
      --alpha 1.0

  # Alpha sweep
  python -m src.train.residual_merge \\
      --base-model google/gemma-4-E4B \\
      --instruct-model google/gemma-4-E4B-it \\
      --cpt-model outputs/cpt_main/final \\
      --alpha 0.5 0.7 0.8 0.9 1.0 1.1 1.2
        """,
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
    args = parser.parse_args()

    if len(args.alpha) == 1:
        compute_residual_merge(
            base_model_id=args.base_model,
            instruct_model_id=args.instruct_model,
            cpt_model_path=args.cpt_model,
            alpha=args.alpha[0],
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        alpha_sweep(
            base_model_id=args.base_model,
            instruct_model_id=args.instruct_model,
            cpt_model_path=args.cpt_model,
            alphas=args.alpha,
            output_dir=args.output_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
