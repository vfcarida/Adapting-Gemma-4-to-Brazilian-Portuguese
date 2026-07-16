"""Supervised Fine-Tuning (SFT) trainer for Portuguese instruction data.

This module wraps TRL's SFTTrainer to fine-tune a model (typically after CPT)
on Portuguese instruction-following data. The pipeline:

1. Load the base model (CPT checkpoint preferred, falls back to original)
2. Apply LoRA adapters for parameter-efficient training
3. Load instruction datasets (Portuguese + optional English)
4. Format conversations using Gemma 4 chat template
5. Train with TRL's SFTTrainer (handles label masking internally)
6. Save final adapter weights (or full model if not using LoRA)

Key Design Decisions:
- SFT trains on the CPT checkpoint (not the IT checkpoint) to add
  instruction-following on top of improved Portuguese knowledge.
- LoRA is default for efficiency; full fine-tuning supported for final runs.
- Uses format_gemma4_chat for strict chat template compliance.
- Think tokens can be optionally included in training data.

Usage:
    python -m src.train.sft_trainer --config configs/train/sft.yaml
"""

import time
from pathlib import Path
from typing import Any

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from src.data.instruction_data_builder import InstructionDataBuilder, format_gemma4_chat
from src.train.callbacks import LocalMetricsCallback, ThroughputCallback
from src.utils.checkpointing import find_latest_checkpoint, save_training_state
from src.utils.config_utils import load_config
from src.utils.hf_utils import load_model_for_training, load_tokenizer
from src.utils.logging_utils import MetricsLogger, get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


class SFTTrainerWrapper:
    """Supervised Fine-Tuning on Portuguese instruction data after CPT.

    This wrapper orchestrates the full SFT pipeline: model loading,
    LoRA setup, data preparation, training, and checkpoint saving.
    It integrates with the project's config system, logging, and
    checkpointing utilities.

    The SFT stage is critical for recovering instruction-following
    capability after CPT (which may degrade it). It uses curated
    Portuguese instruction data distinct from the Aurora-PT corpus
    used during CPT.

    Args:
        config: Full SFT config dict (from configs/train/sft.yaml).

    Attributes:
        train_cfg: Training hyperparameters (lr, batch size, etc.).
        sft_cfg: SFT-specific settings (max_seq_length, packing, think tokens).
        output_dir: Directory for checkpoints and final model.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.seed = config.get("experiment", {}).get("seed", 42)
        self.train_cfg = config["training"]
        self.sft_cfg = config.get("sft", {})
        self.output_dir = Path(config["output"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        """Execute the full SFT pipeline.

        Steps:
        1. Set reproducibility seed
        2. Load model from CPT checkpoint (or base if no checkpoint)
        3. Configure and apply LoRA adapters
        4. Load and format instruction data
        5. Train with SFTTrainer
        6. Save final model and training state

        Returns:
            Dict with training metrics, elapsed time, and config used.
        """
        set_seed(self.seed)
        start_time = time.time()

        # Load model config (may be a path to YAML or inline dict)
        model_cfg = self.config.get("model_config")
        if isinstance(model_cfg, str):
            model_cfg = load_config(model_cfg)

        # Determine base model: prefer CPT checkpoint over original model
        # This ensures SFT builds on Portuguese knowledge from CPT
        base_checkpoint = self.config.get("base_checkpoint")
        if base_checkpoint and Path(base_checkpoint).exists():
            model_id = base_checkpoint
            logger.info(f"Loading from CPT checkpoint: {model_id}")
        else:
            model_id = model_cfg["model"]["base_id"]
            logger.info(f"No CPT checkpoint found, using base: {model_id}")

        # Load tokenizer (same for all Gemma 4 variants)
        tokenizer = load_tokenizer(model_id)

        # When using LoRA, also quantize to 4-bit for memory efficiency
        use_lora = self.train_cfg.get("use_lora", True)
        quantize = use_lora

        model = load_model_for_training(
            model_id,
            use_lora=use_lora,
            quantize=quantize,
            model_config=model_cfg,
        )

        # Apply LoRA adapters for parameter-efficient fine-tuning
        if use_lora:
            # prepare_model_for_kbit_training freezes the base and enables
            # gradient computation only on LoRA parameters
            model = prepare_model_for_kbit_training(model)
            lora_cfg = self.config.get("lora", {})
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 32),
                lora_alpha=lora_cfg.get("lora_alpha", 64),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                # Target all linear layers in the transformer for best results
                target_modules=lora_cfg.get(
                    "target_modules",
                    [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                ),
                task_type="CAUSAL_LM",
                bias="none",
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # Load instruction datasets (Portuguese conversations)
        data_builder = InstructionDataBuilder(self.config)
        instruction_dataset = data_builder.load_datasets()

        # Format conversations using Gemma 4 chat template
        # use_think=True wraps model responses in <think>...</think> tags
        use_think = self.sft_cfg.get("use_think_tokens", False)

        def formatting_func(example):
            """Convert multi-turn messages to Gemma 4 formatted string."""
            messages = example["messages"]
            return format_gemma4_chat(messages, use_think=use_think)

        # Local metrics logger (append-only JSONL for reproducibility)
        metrics_logger = MetricsLogger(
            self.config.get("logging", {}).get("log_file", self.output_dir / "train_log.jsonl")
        )

        # Configure SFTTrainer with all hyperparameters from config
        sft_config = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=self.train_cfg.get("num_train_epochs", 3),
            max_steps=self.train_cfg.get("max_steps", -1),
            per_device_train_batch_size=self.train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=self.train_cfg.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=self.train_cfg["gradient_accumulation_steps"],
            learning_rate=self.train_cfg["learning_rate"],
            lr_scheduler_type=self.train_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.train_cfg.get("warmup_ratio", 0.05),
            weight_decay=self.train_cfg.get("weight_decay", 0.01),
            max_grad_norm=self.train_cfg.get("max_grad_norm", 1.0),
            bf16=self.train_cfg.get("bf16", True),
            # Gradient checkpointing trades compute for memory (essential for large models)
            gradient_checkpointing=self.train_cfg.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs=self.train_cfg.get(
                "gradient_checkpointing_kwargs", {"use_reentrant": False}
            ),
            logging_steps=self.config.get("logging", {}).get("logging_steps", 10),
            save_steps=self.config.get("checkpointing", {}).get("save_steps", 200),
            save_total_limit=self.config.get("checkpointing", {}).get("save_total_limit", 5),
            seed=self.seed,
            # SFT-specific: sequence length and packing
            max_seq_length=self.sft_cfg.get("max_seq_length", 4096),
            packing=self.sft_cfg.get("packing", False),
            report_to=self.config.get("logging", {}).get("report_to", ["none"]),
        )

        # Initialize SFT Trainer with custom callbacks for monitoring
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=instruction_dataset,
            processing_class=tokenizer,
            formatting_func=formatting_func,
            callbacks=[
                ThroughputCallback(metrics_logger),
                LocalMetricsCallback(metrics_logger),
            ],
        )

        # Train with automatic checkpoint resumption
        logger.info("Starting SFT...")
        resume_from = find_latest_checkpoint(self.output_dir)
        train_result = trainer.train(resume_from_checkpoint=resume_from)

        # Save final model (LoRA adapters or full weights)
        final_dir = self.output_dir / "final"
        if use_lora:
            # Save only the LoRA adapter weights (small, ~50-200MB)
            model.save_pretrained(final_dir)
        else:
            # Save full model weights
            trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        # Record training state for reproducibility
        elapsed = time.time() - start_time
        state = {
            "config": self.config,
            "train_result": {
                "global_step": train_result.global_step,
                "training_loss": train_result.training_loss,
                "metrics": train_result.metrics,
            },
            "elapsed_seconds": elapsed,
            "base_checkpoint": str(base_checkpoint),
            "use_lora": use_lora,
        }
        save_training_state(final_dir, state)

        logger.info(f"SFT completed in {elapsed:.1f}s. Model saved to {final_dir}")
        return state


def main():
    """CLI entry point for SFT training."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Supervised Fine-Tuning")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = SFTTrainerWrapper(config)
    trainer.run()


if __name__ == "__main__":
    main()
