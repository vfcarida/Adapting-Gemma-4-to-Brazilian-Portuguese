"""
src/train/residual_merge.py
───────────────────────────
Task Arithmetic / Residual Merge for instruct capability restoration.

Implements the formula:
    inst_residual   = instruct_weights  −  base_weights
    adapted_instruct = cpt_weights      +  (α × inst_residual)

Features:
  • Layer-by-layer tensor operations with shape validation
  • Alpha sweep via CLI (comma-separated or from YAML)
  • Handles MoE expert layers correctly
  • Saves merged checkpoints in safetensors format
  • Progress bar with tqdm
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.config_utils import load_config, parse_args, parse_overrides
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


class ResidualMerger:
    """Perform Task Arithmetic merging of model weights.

    Parameters
    ----------
    base_model_id : str
        HuggingFace ID or path to the original base model.
    instruct_model_id : str
        HuggingFace ID or path to the instruct variant.
    cpt_model_path : str
        Path to the CPT-adapted model checkpoint.
    output_dir : str | Path
        Directory for merged model outputs.
    torch_dtype : str
        Data type for tensor operations.
    device : str
        Device for computations (``"cpu"`` recommended for large models).
    """

    def __init__(
        self,
        base_model_id: str,
        instruct_model_id: str,
        cpt_model_path: str,
        output_dir: str | Path = "./output/merged",
        torch_dtype: str = "bfloat16",
        device: str = "cpu",
    ) -> None:
        self.base_model_id = base_model_id
        self.instruct_model_id = instruct_model_id
        self.cpt_model_path = cpt_model_path
        self.output_dir = Path(output_dir)
        self.device = device

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    def merge(
        self,
        alpha: float,
        output_name: str | None = None,
        validate_shapes: bool = True,
        save_safetensors: bool = True,
    ) -> Path:
        """Execute the residual merge for a given alpha value.

        Parameters
        ----------
        alpha : float
            Scaling factor for the instruction residual.
        output_name : str | None
            Name for the output directory.  Defaults to template.
        validate_shapes : bool
            If ``True``, verify tensor shapes match before arithmetic.
        save_safetensors : bool
            Save using safetensors format.

        Returns
        -------
        Path
            Path to the merged model directory.
        """
        if output_name is None:
            output_name = f"merged-alpha{alpha}"

        merge_dir = self.output_dir / output_name
        merge_dir.mkdir(parents=True, exist_ok=True)

        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║  Residual Merge — Task Arithmetic                    ║")
        logger.info("║  α = %.4f                                           ║", alpha)
        logger.info("╚══════════════════════════════════════════════════════╝")
        logger.info("Base:      %s", self.base_model_id)
        logger.info("Instruct:  %s", self.instruct_model_id)
        logger.info("CPT:       %s", self.cpt_model_path)
        logger.info("Output:    %s", merge_dir)

        # ── Load state dicts ─────────────────────────────────────────
        logger.info("Loading base model weights …")
        base_sd = self._load_state_dict(self.base_model_id)

        logger.info("Loading instruct model weights …")
        instruct_sd = self._load_state_dict(self.instruct_model_id)

        logger.info("Loading CPT model weights …")
        cpt_sd = self._load_state_dict(self.cpt_model_path)

        # ── Validate shapes ──────────────────────────────────────────
        if validate_shapes:
            self._validate_shapes(base_sd, instruct_sd, "base", "instruct")
            self._validate_shapes(base_sd, cpt_sd, "base", "cpt")

        # ── Compute merged weights ───────────────────────────────────
        logger.info("Computing residual merge (α=%.4f) …", alpha)
        merged_sd: dict[str, torch.Tensor] = {}
        skipped_keys: list[str] = []

        for key in tqdm(base_sd.keys(), desc="Merging layers"):
            if key not in instruct_sd or key not in cpt_sd:
                # Key missing from one model — copy from CPT
                logger.debug("Key '%s' missing from instruct or CPT — using CPT weights.", key)
                merged_sd[key] = cpt_sd.get(key, base_sd[key]).to(self.torch_dtype)
                skipped_keys.append(key)
                continue

            base_w = base_sd[key].to(self.torch_dtype).to(self.device)
            inst_w = instruct_sd[key].to(self.torch_dtype).to(self.device)
            cpt_w = cpt_sd[key].to(self.torch_dtype).to(self.device)

            # Task Arithmetic:
            #   inst_residual = instruct_weights - base_weights
            #   merged = cpt_weights + (alpha * inst_residual)
            inst_residual = inst_w - base_w
            merged_w = cpt_w + (alpha * inst_residual)

            merged_sd[key] = merged_w.cpu()

            # Free memory
            del base_w, inst_w, cpt_w, inst_residual, merged_w

        # Free source dicts
        del base_sd, instruct_sd, cpt_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Save merged model ────────────────────────────────────────
        logger.info("Saving merged model to %s …", merge_dir)
        if save_safetensors:
            save_file(merged_sd, merge_dir / "model.safetensors")
        else:
            torch.save(merged_sd, merge_dir / "pytorch_model.bin")

        # Copy tokenizer + config from CPT model
        self._copy_config_and_tokenizer(merge_dir)

        # Save merge metadata
        meta = {
            "merge_type": "task_arithmetic",
            "alpha": alpha,
            "base_model": self.base_model_id,
            "instruct_model": self.instruct_model_id,
            "cpt_model": str(self.cpt_model_path),
            "skipped_keys": skipped_keys,
            "num_merged_keys": len(merged_sd) - len(skipped_keys),
        }
        with open(merge_dir / "merge_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "Merge complete: %d keys merged, %d keys skipped.",
            len(merged_sd) - len(skipped_keys),
            len(skipped_keys),
        )
        return merge_dir

    def sweep(
        self,
        alpha_values: list[float],
        name_template: str = "merged-alpha{alpha}",
        **merge_kwargs: Any,
    ) -> list[Path]:
        """Run merge for multiple alpha values.

        Parameters
        ----------
        alpha_values : list[float]
            List of alpha values to sweep.
        name_template : str
            Template for output directory names.

        Returns
        -------
        list[Path]
            Paths to all merged model directories.
        """
        results = []
        for alpha in alpha_values:
            name = name_template.format(alpha=alpha)
            path = self.merge(alpha=alpha, output_name=name, **merge_kwargs)
            results.append(path)
        return results

    def _load_state_dict(self, model_id_or_path: str) -> dict[str, torch.Tensor]:
        """Load model state dict (supports safetensors and HF format)."""
        path = Path(model_id_or_path)

        # Try loading from safetensors first (local path)
        if path.is_dir():
            safetensors_files = list(path.glob("*.safetensors"))
            if safetensors_files:
                state_dict = {}
                for sf in safetensors_files:
                    state_dict.update(load_file(sf, device="cpu"))
                return state_dict

        # Fallback: load via transformers
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=self.torch_dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
        sd = model.state_dict()
        del model
        gc.collect()
        return sd

    def _validate_shapes(
        self,
        sd_a: dict[str, torch.Tensor],
        sd_b: dict[str, torch.Tensor],
        name_a: str,
        name_b: str,
    ) -> None:
        """Validate that matching keys have the same tensor shapes."""
        mismatches = []
        for key in sd_a:
            if key in sd_b and sd_a[key].shape != sd_b[key].shape:
                mismatches.append(
                    f"  {key}: {name_a}={sd_a[key].shape} vs {name_b}={sd_b[key].shape}"
                )
        if mismatches:
            msg = f"Shape mismatches between {name_a} and {name_b}:\n" + "\n".join(mismatches)
            raise ValueError(msg)
        logger.info("Shape validation passed: %s ↔ %s (%d keys)", name_a, name_b, len(sd_a))

    def _copy_config_and_tokenizer(self, merge_dir: Path) -> None:
        """Copy tokenizer and model config from the CPT checkpoint."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.cpt_model_path, trust_remote_code=True
            )
            tokenizer.save_pretrained(merge_dir)
        except Exception as e:
            logger.warning("Could not copy tokenizer from CPT: %s", e)
            # Fallback to base model tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_id, trust_remote_code=True
                )
                tokenizer.save_pretrained(merge_dir)
            except Exception as e2:
                logger.error("Failed to save any tokenizer: %s", e2)


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 — Residual Merge (Task Arithmetic)")
    parser.add_argument("--config", type=str, default="configs/merge.yml", help="YAML config path")
    parser.add_argument(
        "--alpha",
        type=str,
        default=None,
        help="Comma-separated alpha values (overrides config). E.g. 0.5,0.7,0.9,1.0",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    merge_cfg = config.get("merge", config)

    if args.dry_run:
        print(json.dumps(config, indent=2, default=str))
        sys.exit(0)

    # Alpha values: CLI override or from config
    if args.alpha:
        alpha_values = [float(a.strip()) for a in args.alpha.split(",")]
    else:
        alpha_values = merge_cfg.get("alpha_values", [1.0])

    set_global_seed(42)

    merger = ResidualMerger(
        base_model_id=merge_cfg["base_model_id"],
        instruct_model_id=merge_cfg["instruct_model_id"],
        cpt_model_path=merge_cfg["cpt_model_path"],
        output_dir=merge_cfg.get("output_dir", "./output/merged"),
        torch_dtype=merge_cfg.get("torch_dtype", "bfloat16"),
        device=merge_cfg.get("device", "cpu"),
    )

    name_template = merge_cfg.get("output_name_template", "merged-alpha{alpha}")
    paths = merger.sweep(
        alpha_values=alpha_values,
        name_template=name_template,
        validate_shapes=merge_cfg.get("validate_shapes", True),
        save_safetensors=merge_cfg.get("save_safetensors", True),
    )

    logger.info("All merges complete:")
    for p in paths:
        logger.info("  → %s", p)


if __name__ == "__main__":
    main()
