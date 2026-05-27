"""
src/train/cpt_trainer.py
────────────────────────
Continued Pretraining (CPT) trainer for Gemma 4 with CausalLM.

Uses HuggingFace ``Trainer`` with packed sequences from Aurora-PT.
This is **NOT** SFTTrainer — Aurora-PT is unstructured text and must
be trained with standard next-token prediction (Golden Rule).

Key features:
  • LoRA via PEFT with safe target modules (no multimodal modules)
  • bfloat16 mixed precision
  • Gradient accumulation + checkpointing
  • Dual logging: local JSONL + W&B
  • Fixed seed for reproducibility
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.data.aurora_loader import build_aurora_dataset
from src.data.replay_mix_builder import build_replay_mix_from_config
from src.train.callbacks import JSONLLoggingCallback, PerplexityCallback
from src.utils.config_utils import load_config, parse_args, parse_overrides, validate_config
from src.utils.hf_utils import authenticate_hf, build_lora_config, load_model_and_tokenizer
from src.utils.logging_utils import get_logger, init_wandb, setup_logging
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


class CPTTrainer:
    """Orchestrates continued pretraining for Gemma 4.

    Parameters
    ----------
    config : dict[str, Any]
        Full configuration dict (from YAML).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        validate_config(config, required_keys=["model", "data", "peft", "training"])
        self.config = config
        self.model_cfg = config["model"]
        self.data_cfg = config["data"]
        self.peft_cfg = config["peft"]
        self.train_cfg = config["training"]

    def run(self) -> None:
        """Execute the full CPT pipeline."""
        # ── 1. Seed ──────────────────────────────────────────────────
        seed = self.train_cfg.get("seed", 42)
        set_global_seed(seed)
        logger.info("Global seed set to %d", seed)

        # ── 2. Setup logging ─────────────────────────────────────────
        log_dir = Path(self.train_cfg.get("logging_dir", "reports/training_logs/cpt"))
        setup_logging(log_dir=log_dir)

        # ── 3. W&B ──────────────────────────────────────────────────
        run_name = self.train_cfg.get("run_name", "cpt-gemma4")
        report_to = self.train_cfg.get("report_to", [])
        if "wandb" in report_to:
            import os

            init_wandb(
                project=os.environ.get("WANDB_PROJECT", "gemma4-ptbr-adapt"),
                run_name=run_name,
                config=self.config,
                entity=os.environ.get("WANDB_ENTITY"),
            )

        # ── 4. Authenticate HF ──────────────────────────────────────
        authenticate_hf()

        # ── 5. Load model + tokenizer ────────────────────────────────
        model, tokenizer = load_model_and_tokenizer(
            model_id=self.model_cfg["model_id"],
            torch_dtype=self.model_cfg.get("torch_dtype", "bfloat16"),
            attn_implementation=self.model_cfg.get("attn_implementation", "sdpa"),
            trust_remote_code=self.model_cfg.get("trust_remote_code", True),
        )

        # ── 6. Apply LoRA ────────────────────────────────────────────
        # CRITICAL: Safe target modules only — no multimodal layers
        lora_config = build_lora_config(self.peft_cfg, model=model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ── 7. Build dataset ─────────────────────────────────────────
        # Use replay mix if English ratio is configured, else pure Aurora-PT
        en_ratio = self.data_cfg.get("en_ratio", 0.0)
        if en_ratio > 0:
            logger.info("Building replay mix dataset (PT=%.0f%%, EN=%.0f%%)",
                        (1 - en_ratio) * 100, en_ratio * 100)
            train_dataset = build_replay_mix_from_config(self.config, tokenizer)
        else:
            logger.info("Building pure Aurora-PT dataset")
            train_dataset = build_aurora_dataset(self.config, tokenizer)

        # ── 8. Data collator ─────────────────────────────────────────
        # For packed sequences, labels = input_ids (already set in dataset)
        # We use a simple collator that just stacks tensors
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # CausalLM — no masked LM
        )

        # ── 9. Training arguments ────────────────────────────────────
        output_dir = self.train_cfg.get("output_dir", "./output/cpt")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.train_cfg.get("num_train_epochs", 1),
            max_steps=self.train_cfg.get("max_steps", -1),
            per_device_train_batch_size=self.train_cfg.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=self.train_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=self.train_cfg.get("learning_rate", 2e-4),
            weight_decay=self.train_cfg.get("weight_decay", 0.01),
            warmup_ratio=self.train_cfg.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.train_cfg.get("lr_scheduler_type", "cosine"),
            max_grad_norm=self.train_cfg.get("max_grad_norm", 1.0),
            bf16=self.train_cfg.get("bf16", True),
            fp16=self.train_cfg.get("fp16", False),
            gradient_checkpointing=self.train_cfg.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs=self.train_cfg.get(
                "gradient_checkpointing_kwargs", {"use_reentrant": False}
            ),
            logging_steps=self.train_cfg.get("logging_steps", 10),
            logging_dir=str(log_dir),
            report_to=report_to,
            run_name=run_name,
            save_strategy=self.train_cfg.get("save_strategy", "steps"),
            save_steps=self.train_cfg.get("save_steps", 2500),
            save_total_limit=self.train_cfg.get("save_total_limit", 5),
            seed=seed,
            dataloader_num_workers=self.train_cfg.get("dataloader_num_workers", 4),
            dataloader_pin_memory=self.train_cfg.get("dataloader_pin_memory", True),
            remove_unused_columns=False,  # Packed sequences have custom columns
            ddp_find_unused_parameters=self.train_cfg.get("ddp_find_unused_parameters", False),
            deepspeed=self.train_cfg.get("deepspeed"),
        )

        # ── 10. Custom callbacks ─────────────────────────────────────
        callbacks = [
            JSONLLoggingCallback(log_dir=str(log_dir)),
        ]

        # ── 11. Trainer ──────────────────────────────────────────────
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # ── 12. Train! ───────────────────────────────────────────────
        logger.info("Starting continued pretraining …")
        trainer.train()

        # ── 13. Save final checkpoint ────────────────────────────────
        final_dir = Path(output_dir) / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("Training complete. Final checkpoint → %s", final_dir)


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args(description="Gemma 4 — Continued Pretraining (CPT)")
    overrides = parse_overrides(args.override) if args.override else {}
    config = load_config(args.config, overrides=overrides)

    if args.dry_run:
        import json

        print(json.dumps(config, indent=2, default=str))
        sys.exit(0)

    trainer = CPTTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
