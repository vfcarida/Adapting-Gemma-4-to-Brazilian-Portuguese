"""Continued Pretraining trainer for causal language modeling."""

import time
from pathlib import Path
from typing import Any

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.data.aurora_loader import AuroraLoader, tokenize_for_cpt
from src.data.replay_mix_builder import ReplayMixBuilder
from src.train.callbacks import LocalMetricsCallback, ThroughputCallback
from src.utils.checkpointing import find_latest_checkpoint, save_training_state
from src.utils.config_utils import load_config
from src.utils.hf_utils import load_model_for_training, load_tokenizer
from src.utils.logging_utils import MetricsLogger, get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


class CPTTrainer:
    """Continued Pretraining on Portuguese corpus."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.seed = config.get("experiment", {}).get("seed", 42)
        self.train_cfg = config["training"]
        self.output_dir = Path(config["output"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        """Execute full CPT pipeline."""
        set_seed(self.seed)
        start_time = time.time()

        # Load model config
        model_cfg = config if isinstance(
            (config := self.config.get("model_config")), dict
        ) else load_config(config)

        # Load data config
        data_cfg = self.config.get("data_config")
        if isinstance(data_cfg, str):
            data_cfg = load_config(data_cfg)

        # Load tokenizer and model
        model_id = model_cfg["model"]["base_id"]
        tokenizer = load_tokenizer(model_id)

        use_lora = self.train_cfg.get("use_lora", False)
        quantize = use_lora  # Quantize only when using LoRA

        model = load_model_for_training(
            model_id,
            use_lora=use_lora,
            quantize=quantize,
            model_config=model_cfg,
        )

        # Apply LoRA if configured
        if use_lora:
            model = prepare_model_for_kbit_training(model)
            lora_cfg = self.config.get("lora", {})
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 64),
                lora_alpha=lora_cfg.get("lora_alpha", 128),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                target_modules=lora_cfg.get("target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]),
                task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
                bias=lora_cfg.get("bias", "none"),
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # Load and prepare data
        logger.info("Loading and preparing training data...")
        aurora_loader = AuroraLoader(data_cfg)
        splits = aurora_loader.load_and_prepare()

        # Build mixture if configured
        mixture_name = self.config.get("data_mixture", "pt_only")
        if mixture_name != "pt_only":
            mix_builder = ReplayMixBuilder(data_cfg)
            train_dataset = mix_builder.build_mixture(mixture_name, splits["train"])
        else:
            train_dataset = splits["train"]

        # Tokenize and pack
        max_seq_length = model_cfg["model"].get("max_seq_length", 8192)
        pack = data_cfg.get("packing", {}).get("enabled", True)

        train_tokenized = tokenize_for_cpt(
            train_dataset, tokenizer, max_seq_length=max_seq_length, pack=pack
        )
        val_tokenized = tokenize_for_cpt(
            splits["validation"], tokenizer, max_seq_length=max_seq_length, pack=pack
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=self.config["output"].get("overwrite_output_dir", False),
            num_train_epochs=self.train_cfg.get("num_train_epochs", 1),
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
            tf32=self.train_cfg.get("tf32", True),
            gradient_checkpointing=self.train_cfg.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs=self.train_cfg.get(
                "gradient_checkpointing_kwargs", {"use_reentrant": False}
            ),
            dataloader_num_workers=self.train_cfg.get("dataloader_num_workers", 4),
            dataloader_pin_memory=self.train_cfg.get("dataloader_pin_memory", True),
            logging_steps=self.config.get("logging", {}).get("logging_steps", 10),
            save_steps=self.config.get("checkpointing", {}).get("save_steps", 500),
            save_total_limit=self.config.get("checkpointing", {}).get("save_total_limit", 5),
            eval_strategy=self.config.get("evaluation", {}).get("eval_strategy", "steps"),
            eval_steps=self.config.get("evaluation", {}).get("eval_steps", 500),
            load_best_model_at_end=True,
            metric_for_best_model=self.config.get("evaluation", {}).get(
                "metric_for_best_model", "eval_loss"
            ),
            greater_is_better=False,
            report_to=self.config.get("logging", {}).get("report_to", ["none"]),
            seed=self.seed,
            data_seed=self.seed,
            run_name=self.config.get("experiment", {}).get("name", "cpt"),
        )

        # Check for resume
        resume_from = self.config.get("checkpointing", {}).get("resume_from_checkpoint")
        if resume_from is None:
            resume_from = find_latest_checkpoint(self.output_dir)

        # Callbacks
        metrics_logger = MetricsLogger(
            self.config.get("logging", {}).get("log_file", self.output_dir / "train_log.jsonl")
        )
        callbacks = [
            ThroughputCallback(metrics_logger),
            LocalMetricsCallback(metrics_logger),
        ]

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Train
        logger.info("Starting continued pretraining...")
        train_result = trainer.train(resume_from_checkpoint=resume_from)

        # Save final model
        final_dir = self.output_dir / "final"
        if use_lora:
            model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
        else:
            trainer.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)

        # Save training state
        elapsed = time.time() - start_time
        state = {
            "config": self.config,
            "train_result": {
                "global_step": train_result.global_step,
                "training_loss": train_result.training_loss,
                "metrics": train_result.metrics,
            },
            "elapsed_seconds": elapsed,
            "model_id": model_id,
            "use_lora": use_lora,
        }
        save_training_state(final_dir, state)

        logger.info(f"CPT completed in {elapsed:.1f}s. Model saved to {final_dir}")
        return state


def main():
    """CLI entry point for CPT."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Continued Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--override", nargs="*", help="Override config values (key=value)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides
    if args.override:
        from src.utils.config_utils import merge_configs
        overrides = {}
        for o in args.override:
            key, value = o.split("=", 1)
            keys = key.split(".")
            d = overrides
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        config = merge_configs(config, overrides)

    trainer = CPTTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
