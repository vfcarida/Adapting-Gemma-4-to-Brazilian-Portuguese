"""DPO (Direct Preference Optimization) trainer — optional alignment stage.

This module implements preference tuning after SFT using TRL's DPOTrainer.
DPO is an optional stage that can further improve model quality by learning
from human preference pairs (chosen vs rejected responses).

Pipeline position: CPT → SFT → DPO (optional)

When to use DPO:
- When you have preference data for Portuguese instruction following
- When SFT alone shows issues like verbosity, hallucination, or unsafe outputs
- When the model needs to align more closely with human preferences

The DPO loss directly optimizes the policy to prefer chosen over rejected
responses without needing a separate reward model (unlike RLHF).

Supported loss types:
- "sigmoid" (default): Standard DPO loss from Rafailov et al. 2023
- "hinge": Margin-based loss variant
- "ipo": Identity Preference Optimization

Usage:
    python -m src.train.dpo_trainer --config configs/train/dpo.yaml
"""

import time
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

from src.data.instruction_data_builder import format_gemma4_chat
from src.train.callbacks import LocalMetricsCallback
from src.utils.checkpointing import save_training_state
from src.utils.config_utils import load_config
from src.utils.hf_utils import load_model_for_training, load_tokenizer
from src.utils.logging_utils import MetricsLogger, get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


class DPOTrainerWrapper:
    """DPO preference tuning after SFT.

    This wrapper handles the complete DPO pipeline: loading the SFT
    checkpoint, preparing preference data in the correct format, and
    running DPO training with TRL.

    The preference data must contain:
    - prompt: The user's question/instruction
    - chosen: The preferred (better) response
    - rejected: The dispreferred (worse) response

    Args:
        config: Full DPO config dict (from configs/train/dpo.yaml).

    Attributes:
        dpo_cfg: DPO-specific settings (beta, loss_type, max lengths).
        train_cfg: General training hyperparameters.
        output_dir: Directory for checkpoints and final model.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.seed = config.get("experiment", {}).get("seed", 42)
        self.train_cfg = config["training"]
        self.dpo_cfg = config.get("dpo", {})
        self.output_dir = Path(config["output"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_preference_data(self) -> Dataset:
        """Load and format preference dataset for DPO.

        Supports loading from:
        - HuggingFace Hub (dpo.dataset_hub_id)
        - Local JSONL file (dpo.dataset_path)

        Each example is formatted with the Gemma 4 chat template so the
        DPO trainer sees properly templated prompts.

        Returns:
            HuggingFace Dataset with columns: prompt, chosen, rejected.

        Raises:
            FileNotFoundError: If neither hub_id nor local path is configured.
        """
        hub_id = self.dpo_cfg.get("dataset_hub_id")
        local_path = self.dpo_cfg.get("dataset_path")

        if hub_id:
            ds = load_dataset(hub_id, split="train")
        elif local_path and Path(local_path).exists():
            ds = load_dataset("json", data_files=local_path, split="train")
        else:
            raise FileNotFoundError(
                "No preference dataset found. Set dpo.dataset_hub_id or dpo.dataset_path"
            )

        # Format prompts with Gemma 4 chat template
        # DPO needs: formatted prompt + raw chosen/rejected completions
        def format_example(example):
            prompt = example.get("prompt", "")
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")

            # Apply chat template to the prompt (add_generation_prompt=True
            # means the template ends with "model\n" ready for completion)
            prompt_formatted = format_gemma4_chat(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
            return {
                "prompt": prompt_formatted,
                "chosen": chosen,
                "rejected": rejected,
            }

        return ds.map(format_example)

    def run(self) -> dict[str, Any]:
        """Execute the full DPO training pipeline.

        Steps:
        1. Load SFT checkpoint (or base model)
        2. Apply LoRA for parameter-efficient alignment
        3. Load and format preference data
        4. Train with DPOTrainer
        5. Save aligned model

        Returns:
            Dict with training metrics and elapsed time.
        """
        set_seed(self.seed)
        start_time = time.time()

        # Load model configuration
        model_cfg = self.config.get("model_config")
        if isinstance(model_cfg, str):
            model_cfg = load_config(model_cfg)

        # Use SFT checkpoint as base for DPO (pipeline: CPT → SFT → DPO)
        base_checkpoint = self.config.get("base_checkpoint")
        model_id = (
            base_checkpoint
            if base_checkpoint and Path(base_checkpoint).exists()
            else model_cfg["model"]["base_id"]
        )

        tokenizer = load_tokenizer(model_id)
        use_lora = self.train_cfg.get("use_lora", True)

        model = load_model_for_training(
            model_id, use_lora=use_lora, quantize=use_lora, model_config=model_cfg
        )

        # Apply LoRA (smaller rank than SFT since DPO is a refinement stage)
        if use_lora:
            model = prepare_model_for_kbit_training(model)
            lora_cfg = self.config.get("lora", {})
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("lora_alpha", 32),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                # Only attention layers for DPO (lighter than SFT)
                target_modules=lora_cfg.get(
                    "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
                ),
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

        # Load preference data
        dataset = self._load_preference_data()

        # Local metrics logging
        metrics_logger = MetricsLogger(self.output_dir / "train_log.jsonl")

        # DPO training configuration
        # beta controls the KL penalty strength (lower = more deviation from reference)
        dpo_config = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=self.train_cfg.get("num_train_epochs", 1),
            max_steps=self.train_cfg.get("max_steps", 2000),
            per_device_train_batch_size=self.train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=self.train_cfg["gradient_accumulation_steps"],
            learning_rate=self.train_cfg["learning_rate"],
            lr_scheduler_type=self.train_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.train_cfg.get("warmup_ratio", 0.1),
            bf16=self.train_cfg.get("bf16", True),
            gradient_checkpointing=self.train_cfg.get("gradient_checkpointing", True),
            # DPO-specific parameters
            beta=self.dpo_cfg.get("beta", 0.1),  # KL divergence coefficient
            loss_type=self.dpo_cfg.get("loss_type", "sigmoid"),  # DPO loss variant
            max_prompt_length=self.dpo_cfg.get("max_prompt_length", 1024),
            max_length=self.dpo_cfg.get("max_length", 2048),
            logging_steps=10,
            save_steps=self.config.get("checkpointing", {}).get("save_steps", 500),
            seed=self.seed,
            report_to=["none"],
        )

        # Initialize DPO trainer
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[LocalMetricsCallback(metrics_logger)],
        )

        # Train
        logger.info("Starting DPO training...")
        train_result = trainer.train()

        # Save final aligned model
        final_dir = self.output_dir / "final"
        if use_lora:
            model.save_pretrained(final_dir)
        else:
            trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        # Record training state
        elapsed = time.time() - start_time
        state = {
            "elapsed_seconds": elapsed,
            "global_step": train_result.global_step,
            "training_loss": train_result.training_loss,
        }
        save_training_state(final_dir, state)
        logger.info(f"DPO completed in {elapsed:.1f}s")
        return state


def main():
    """CLI entry point for DPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="Run DPO Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = DPOTrainerWrapper(config)
    trainer.run()


if __name__ == "__main__":
    main()
