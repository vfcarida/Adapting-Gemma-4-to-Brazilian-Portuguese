"""
src/train/dpo_trainer.py
────────────────────────
Direct Preference Optimization (DPO) using TRL's ``DPOTrainer``.

Optional Stage 5 in the adaptation pipeline:
  Base -> CPT -> SFT -> DPO

Features:
  • Gemma 4 chat template integration
  • LoRA with safe target modules
  • Configurable via YAML
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import get_peft_model
from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer

from src.train.callbacks import JSONLLoggingCallback
from src.utils.config_utils import load_config, parse_args, parse_overrides, validate_config
from src.utils.hf_utils import authenticate_hf, build_lora_config, load_model_and_tokenizer
from src.utils.logging_utils import get_logger, init_wandb, setup_logging
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


class DPOTrainerWrapper:
    """Wraps TRL DPOTrainer for Gemma 4 preference tuning.

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
        """Execute the DPO pipeline."""
        seed = self.train_cfg.get("seed", 42)
        set_global_seed(seed)

        log_dir = Path(self.train_cfg.get("logging_dir", "reports/training_logs/dpo"))
        setup_logging(log_dir=log_dir)

        run_name = self.train_cfg.get("run_name", "dpo-gemma4-ptbr")
        report_to = self.train_cfg.get("report_to", [])
        if "wandb" in report_to:
            init_wandb(
                project=os.environ.get("WANDB_PROJECT", "gemma4-ptbr-adapt"),
                run_name=run_name,
                config=self.config,
                entity=os.environ.get("WANDB_ENTITY"),
            )

        authenticate_hf()

        # Load reference and active models
        model, tokenizer = load_model_and_tokenizer(
            model_id=self.model_cfg["model_id"],
            torch_dtype=self.model_cfg.get("torch_dtype", "bfloat16"),
            attn_implementation=self.model_cfg.get("attn_implementation", "sdpa"),
        )
        
        # In TRL DPO, if we pass peft_config, it automatically creates the reference model
        lora_config = build_lora_config(self.peft_cfg, model=model)
        
        # Load dataset
        dataset_id = self.data_cfg["dataset_id"]
        if dataset_id == "SET_YOUR_PREFERENCE_DATASET_HERE":
            raise ValueError(
                "DPO dataset not configured. Edit configs/dpo.yml → data.dataset_id "
                "with a valid preference dataset (must have prompt, chosen, rejected)."
            )

        from datasets import load_dataset
        train_ds = load_dataset(dataset_id, split=self.data_cfg.get("dataset_split", "train"))
        
        # Ensure correct column names: prompt, chosen, rejected
        # which DPOTrainer expects by default.
        if "prompt_column" in self.data_cfg:
            train_ds = train_ds.rename_column(self.data_cfg["prompt_column"], "prompt")
        if "chosen_column" in self.data_cfg:
            train_ds = train_ds.rename_column(self.data_cfg["chosen_column"], "chosen")
        if "rejected_column" in self.data_cfg:
            train_ds = train_ds.rename_column(self.data_cfg["rejected_column"], "rejected")

        # Format dataset using Gemma chat template if they aren't already formatted
        def apply_template(example):
            prompt_msgs = [{"role": "user", "content": example["prompt"]}]
            chosen_msgs = [{"role": "assistant", "content": example["chosen"]}]
            rejected_msgs = [{"role": "assistant", "content": example["rejected"]}]
            
            example["prompt"] = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            # DPO expects chosen and rejected to be the raw text of the response (without the prompt)
            example["chosen"] = chosen_msgs[0]["content"]
            example["rejected"] = rejected_msgs[0]["content"]
            return example
            
        train_ds = train_ds.map(apply_template)

        val_size = self.data_cfg.get("val_size", 0.05)
        split = train_ds.train_test_split(test_size=val_size, seed=seed)
        train_ds = split["train"]
        val_ds = split["test"]

        output_dir = self.train_cfg.get("output_dir", "./output/dpo")

        dpo_config = DPOConfig(
            output_dir=output_dir,
            beta=self.train_cfg.get("beta", 0.1),
            num_train_epochs=self.train_cfg.get("num_train_epochs", 1),
            max_steps=self.train_cfg.get("max_steps", -1),
            per_device_train_batch_size=self.train_cfg.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=self.train_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=self.train_cfg.get("learning_rate", 5e-6), # Lower LR for DPO
            weight_decay=self.train_cfg.get("weight_decay", 0.01),
            warmup_ratio=self.train_cfg.get("warmup_ratio", 0.1),
            lr_scheduler_type=self.train_cfg.get("lr_scheduler_type", "cosine"),
            bf16=self.train_cfg.get("bf16", True),
            gradient_checkpointing=self.train_cfg.get("gradient_checkpointing", True),
            logging_steps=self.train_cfg.get("logging_steps", 5),
            logging_dir=str(log_dir),
            report_to=report_to,
            run_name=run_name,
            save_strategy=self.train_cfg.get("save_strategy", "epoch"),
            eval_strategy=self.train_cfg.get("eval_strategy", "epoch"),
            seed=seed,
            max_length=self.data_cfg.get("max_length", 2048),
            max_prompt_length=self.data_cfg.get("max_prompt_length", 1024),
            remove_unused_columns=False,
        )

        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            peft_config=lora_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            callbacks=[JSONLLoggingCallback(log_dir=str(log_dir))],
        )

        logger.info("Starting DPO …")
        trainer.train()

        final_dir = Path(output_dir) / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("DPO complete. Final checkpoint → %s", final_dir)

def main() -> None:
    args = parse_args(description="Gemma 4 — Direct Preference Optimization (DPO)")
    overrides = parse_overrides(args.override) if args.override else {}
    config = load_config(args.config, overrides=overrides)

    if args.dry_run:
        import json
        print(json.dumps(config, indent=2, default=str))
        sys.exit(0)

    wrapper = DPOTrainerWrapper(config)
    wrapper.run()

if __name__ == "__main__":
    main()
