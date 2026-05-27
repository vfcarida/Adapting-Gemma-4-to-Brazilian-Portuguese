"""
src/train/sft_trainer.py
────────────────────────
Supervised Fine-Tuning (SFT) using TRL's ``SFTTrainer`` with Gemma 4
official chat template.

This module is the **only** place where ``SFTTrainer`` is used — and
only with structured instruction data (human/assistant pairs), NEVER
with raw Aurora-PT text (Golden Rule).

Features:
  • Gemma 4 chat template integration
  • Label masking: loss only on assistant turns
  • LoRA with safe target modules
  • Configurable via YAML
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from peft import get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer

from src.data.instruction_data_builder import InstructionDataBuilder
from src.train.callbacks import JSONLLoggingCallback
from src.utils.config_utils import load_config, parse_args, parse_overrides, validate_config
from src.utils.hf_utils import authenticate_hf, build_lora_config, load_model_and_tokenizer
from src.utils.logging_utils import get_logger, init_wandb, setup_logging
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


class SFTTrainerWrapper:
    """Wraps TRL SFTTrainer for Gemma 4 instruction fine-tuning.

    Parameters
    ----------
    config : dict[str, Any]
        Full YAML configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        validate_config(config, required_keys=["model", "data", "peft", "training"])
        self.config = config
        self.model_cfg = config["model"]
        self.data_cfg = config["data"]
        self.peft_cfg = config["peft"]
        self.train_cfg = config["training"]

    def run(self) -> None:
        """Execute the SFT pipeline."""
        # ── 1. Seed ──────────────────────────────────────────────────
        seed = self.train_cfg.get("seed", 42)
        set_global_seed(seed)

        # ── 2. Logging ───────────────────────────────────────────────
        log_dir = Path(self.train_cfg.get("logging_dir", "reports/training_logs/sft"))
        setup_logging(log_dir=log_dir)

        # ── 3. W&B ──────────────────────────────────────────────────
        run_name = self.train_cfg.get("run_name", "sft-gemma4-ptbr")
        report_to = self.train_cfg.get("report_to", [])
        if "wandb" in report_to:
            init_wandb(
                project=os.environ.get("WANDB_PROJECT", "gemma4-ptbr-adapt"),
                run_name=run_name,
                config=self.config,
                entity=os.environ.get("WANDB_ENTITY"),
            )

        # ── 4. Auth ──────────────────────────────────────────────────
        authenticate_hf()

        # ── 5. Model + tokenizer ─────────────────────────────────────
        model, tokenizer = load_model_and_tokenizer(
            model_id=self.model_cfg["model_id"],
            torch_dtype=self.model_cfg.get("torch_dtype", "bfloat16"),
            attn_implementation=self.model_cfg.get("attn_implementation", "sdpa"),
        )

        # ── 6. LoRA ──────────────────────────────────────────────────
        lora_config = build_lora_config(self.peft_cfg, model=model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ── 7. Dataset ───────────────────────────────────────────────
        dataset_id = self.data_cfg["dataset_id"]
        if dataset_id == "SET_YOUR_INSTRUCTION_DATASET_HERE":
            raise ValueError(
                "SFT dataset not configured. Edit configs/sft.yml → data.dataset_id "
                "with a valid instruction dataset (e.g. a PT-BR chat dataset). "
                "NEVER use Aurora-PT raw text here (Golden Rule)."
            )

        # Load train and validation splits
        train_ds = self._load_and_format_dataset(tokenizer, split=self.data_cfg.get("dataset_split", "train"))

        val_ds = None
        val_split = self.data_cfg.get("val_split")
        if val_split:
            try:
                val_ds = self._load_and_format_dataset(tokenizer, split=val_split)
            except Exception:
                logger.warning("Validation split '%s' not found — auto-splitting.", val_split)

        # Auto-split if no validation set
        if val_ds is None:
            val_size = self.data_cfg.get("val_size", 0.05)
            split = train_ds.train_test_split(test_size=val_size, seed=seed)
            train_ds = split["train"]
            val_ds = split["test"]
            logger.info("Auto-split: train=%d, val=%d", len(train_ds), len(val_ds))

        # ── 8. SFT Config ───────────────────────────────────────────
        output_dir = self.train_cfg.get("output_dir", "./output/sft")

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.train_cfg.get("num_train_epochs", 3),
            max_steps=self.train_cfg.get("max_steps", -1),
            per_device_train_batch_size=self.train_cfg.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=self.train_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=self.train_cfg.get("learning_rate", 5e-5),
            weight_decay=self.train_cfg.get("weight_decay", 0.01),
            warmup_ratio=self.train_cfg.get("warmup_ratio", 0.05),
            lr_scheduler_type=self.train_cfg.get("lr_scheduler_type", "cosine"),
            max_grad_norm=self.train_cfg.get("max_grad_norm", 1.0),
            bf16=self.train_cfg.get("bf16", True),
            fp16=self.train_cfg.get("fp16", False),
            gradient_checkpointing=self.train_cfg.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs=self.train_cfg.get(
                "gradient_checkpointing_kwargs", {"use_reentrant": False}
            ),
            logging_steps=self.train_cfg.get("logging_steps", 5),
            logging_dir=str(log_dir),
            report_to=report_to,
            run_name=run_name,
            save_strategy=self.train_cfg.get("save_strategy", "epoch"),
            save_total_limit=self.train_cfg.get("save_total_limit", 3),
            load_best_model_at_end=self.train_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=self.train_cfg.get("metric_for_best_model", "eval_loss"),
            greater_is_better=self.train_cfg.get("greater_is_better", False),
            eval_strategy=self.train_cfg.get("eval_strategy", "epoch"),
            per_device_eval_batch_size=self.train_cfg.get("per_device_eval_batch_size", 4),
            seed=seed,
            max_seq_length=self.data_cfg.get("max_seq_len", 4096),
            dataset_text_field=None,  # We provide pre-tokenized data
            remove_unused_columns=False,
        )

        # ── 9. Trainer ───────────────────────────────────────────────
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            callbacks=[JSONLLoggingCallback(log_dir=str(log_dir))],
        )

        # ── 10. Train ────────────────────────────────────────────────
        logger.info("Starting SFT …")
        trainer.train()

        # ── 11. Save ─────────────────────────────────────────────────
        final_dir = Path(output_dir) / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("SFT complete. Final checkpoint → %s", final_dir)

    def _load_and_format_dataset(
        self,
        tokenizer: AutoTokenizer,
        split: str,
    ) -> Dataset:
        """Load and format the instruction dataset."""
        builder = InstructionDataBuilder(
            tokenizer=tokenizer,
            max_seq_len=self.data_cfg.get("max_seq_len", 4096),
        )
        return builder.build_dataset(
            dataset_id=self.data_cfg["dataset_id"],
            split=split,
            messages_column=self.data_cfg.get("messages_column"),
            human_column=self.data_cfg.get("human_column"),
            assistant_column=self.data_cfg.get("assistant_column"),
            system_column=self.data_cfg.get("system_column"),
        )


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args(description="Gemma 4 — Supervised Fine-Tuning (SFT)")
    overrides = parse_overrides(args.override) if args.override else {}
    config = load_config(args.config, overrides=overrides)

    if args.dry_run:
        import json

        print(json.dumps(config, indent=2, default=str))
        sys.exit(0)

    wrapper = SFTTrainerWrapper(config)
    wrapper.run()


if __name__ == "__main__":
    main()
