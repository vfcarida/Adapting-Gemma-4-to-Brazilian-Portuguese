"""Continued Pretraining trainer for causal language modeling.

Supports:
- Single GPU (LoRA/QLoRA)
- Multi-GPU via DeepSpeed ZeRO-2/3 (full fine-tune)
- Spot instance preemption recovery (auto-resume from latest checkpoint)
- GCS checkpoint sync via callback
"""

import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.data.aurora_loader import AuroraLoader, tokenize_for_cpt
from src.data.replay_mix_builder import ReplayMixBuilder
from src.train.callbacks import (
    ForgettingMonitorCallback,
    LocalMetricsCallback,
    ThroughputCallback,
    WandBCallback,
)
from src.train.peft_factories import create_peft_config
from src.utils.checkpointing import find_latest_checkpoint, save_training_state
from src.utils.config_utils import load_config
from src.utils.hf_utils import load_model_for_training, load_tokenizer
from src.utils.logging_utils import MetricsLogger, get_logger, resolve_report_to
from src.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class PackedSequenceCollator:
    """Collate pre-packed, pre-labeled CPT sequences without touching labels.

    `pack_sequences` (src/data/aurora_loader.py) already produces
    fixed-length `input_ids`/`labels` pairs with EOS separators and optional
    cross-document `-100` masking baked in. `DataCollatorForLanguageModeling
    (mlm=False)` unconditionally *recomputes* `labels = input_ids.clone()`
    and masks `pad_token_id` positions, discarding whatever label masking
    `pack_sequences` did. Since every packed sequence is already exactly
    `max_seq_length` (no padding involved), the correct collator here is a
    simple tensor stack — no padding, no label recomputation.
    """

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


class PreemptionHandler(TrainerCallback):
    """Handle Spot/Preemptible VM preemption gracefully.

    GCP sends SIGTERM 30 seconds before terminating a Spot VM.
    This callback catches the signal, triggers an immediate checkpoint save,
    and stops training cleanly so it can be resumed later.
    """

    def __init__(self):
        self._preempted = False
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def _handle_sigterm(self, signum, frame):
        logger.warning("SIGTERM received — Spot preemption detected. Saving checkpoint...")
        self._preempted = True

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if self._preempted:
            control.should_save = True
            control.should_training_stop = True
            logger.warning(f"Preemption: saving at step {state.global_step} and stopping.")
        return control


class GCSCheckpointSync(TrainerCallback):
    """Sync checkpoints to GCS after each save for fault tolerance.

    Only activates if GCS_BUCKET environment variable is set.
    Runs gsutil rsync in the background to avoid blocking training,
    but verifies previous sync completed before starting a new one.

    Features:
    - Skips new sync if previous is still running (avoids parallel rsyncs)
    - Logs sync failures as warnings
    - Waits for final sync on training end
    """

    def __init__(self, output_dir: str):
        import os

        self.bucket = os.environ.get("GCS_BUCKET")
        self.output_dir = output_dir
        self.experiment_name = Path(output_dir).name
        self._last_process = None
        self._sync_count = 0
        self._fail_count = 0

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if not self.bucket or not state.is_world_process_zero:
            return
        import subprocess

        # Check if previous sync is still running
        if self._last_process is not None:
            poll = self._last_process.poll()
            if poll is None:
                logger.warning("GCS sync: previous sync still running, skipping this save")
                return
            elif poll != 0:
                self._fail_count += 1
                logger.warning(f"GCS sync: previous sync failed (exit={poll})")

        if not Path(self.output_dir).exists():
            logger.warning(f"GCS sync: output dir does not exist: {self.output_dir}")
            return

        gcs_path = f"{self.bucket}/outputs/{self.experiment_name}/"
        cmd = f"gsutil -m rsync -r {self.output_dir}/ {gcs_path}"
        logger.info(f"Syncing checkpoint to {gcs_path} (sync #{self._sync_count + 1})")
        self._last_process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self._sync_count += 1

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Wait for final sync to complete before training ends."""
        if not state.is_world_process_zero:
            return
        if self._last_process is not None and self._last_process.poll() is None:
            logger.info("GCS sync: waiting for final sync to complete...")
            self._last_process.wait(timeout=300)
        if self._sync_count > 0:
            logger.info(f"GCS sync summary: {self._sync_count} syncs, {self._fail_count} failures")


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
        model_cfg = self.config.get("model_config")
        if model_cfg is None:
            raise ValueError("model_config is required in training config")
        if isinstance(model_cfg, str):
            model_cfg = load_config(model_cfg)

        # Load data config
        data_cfg = self.config.get("data_config")
        if isinstance(data_cfg, str):
            data_cfg = load_config(data_cfg)

        model_id = model_cfg["model"]["base_id"]
        use_lora = self.train_cfg.get("use_lora", False)
        quantize = use_lora  # Quantize only when using LoRA

        # --- Resolve DeepSpeed config path (if specified) ---
        deepspeed_config = self.train_cfg.get("deepspeed") or self.config.get("deepspeed")
        if deepspeed_config and isinstance(deepspeed_config, str):
            ds_path = Path(deepspeed_config)
            if not ds_path.is_absolute():
                ds_path = Path.cwd() / ds_path
            if ds_path.exists():
                deepspeed_config = str(ds_path)
                logger.info(f"DeepSpeed enabled: {deepspeed_config}")
            else:
                logger.warning(
                    f"DeepSpeed config not found at {ds_path} — training will run WITHOUT "
                    "DeepSpeed. If this is a multi-GPU ZeRO-3 run, fix the path before continuing."
                )
                deepspeed_config = None

        # --- Resolve save/eval step compatibility BEFORE building TrainingArguments ---
        # transformers requires save_steps to be a round multiple of eval_steps when
        # load_best_model_at_end=True and both strategies are "steps". Rather than
        # crash on a misconfigured YAML (this repo's default pilot/lr_sweep configs
        # historically violated this), degrade gracefully: disable
        # load_best_model_at_end (fall back to "always keep the last checkpoint")
        # and warn loudly instead of raising at TrainingArguments construction time.
        eval_cfg = self.config.get("evaluation", {})
        checkpoint_cfg = self.config.get("checkpointing", {})
        save_steps = checkpoint_cfg.get("save_steps", 500)
        eval_steps = eval_cfg.get("eval_steps", 500)
        eval_strategy = eval_cfg.get("eval_strategy", "steps")
        load_best_model_at_end = eval_cfg.get("load_best_model_at_end", True)
        if load_best_model_at_end and eval_strategy == "steps" and save_steps % eval_steps != 0:
            logger.warning(
                f"checkpointing.save_steps={save_steps} is not a multiple of "
                f"evaluation.eval_steps={eval_steps} — transformers requires this when "
                "load_best_model_at_end=True. Disabling load_best_model_at_end for this "
                "run (checkpoints are still saved every save_steps; the LAST one, not "
                "necessarily the best, is used as 'final')."
            )
            load_best_model_at_end = False

        # --- Build TrainingArguments BEFORE loading the model ---
        # This ordering matters for DeepSpeed ZeRO-3: passing `deepspeed=...` here
        # constructs transformers' global HfDeepSpeedConfig, which is what makes
        # `from_pretrained` shard the model at load time via `zero.Init`. If the
        # model were loaded first (the previous ordering), every rank would
        # materialize the FULL model in memory before ZeRO-3 ever gets a chance
        # to shard it.
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
            save_steps=save_steps,
            save_total_limit=checkpoint_cfg.get("save_total_limit", 5),
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=eval_cfg.get("metric_for_best_model", "eval_loss"),
            greater_is_better=eval_cfg.get("greater_is_better", False),
            report_to=resolve_report_to(self.config.get("logging", {}).get("report_to")),
            seed=self.seed,
            data_seed=self.seed,
            run_name=self.config.get("experiment", {}).get("name", "cpt"),
            include_num_input_tokens_seen=True,
            # DeepSpeed integration
            deepspeed=deepspeed_config,
            # HF Hub checkpoint persistence — the recommended pattern for
            # ephemeral single-GPU sessions (Colab) where local disk doesn't
            # survive a session restart, as an alternative/complement to the
            # GCS sync path (GCSCheckpointSync below) used on GCP VMs. Set
            # output.push_to_hub: true + output.hub_model_id in the training
            # config to enable; hub_strategy="checkpoint" pushes to a
            # `last-checkpoint/` subfolder on every save, which a new
            # session can download and resume from via
            # `snapshot_download(repo_id, allow_patterns="last-checkpoint/*")`.
            push_to_hub=self.config["output"].get("push_to_hub", False),
            hub_model_id=self.config["output"].get("hub_model_id"),
            hub_strategy=self.config["output"].get("hub_strategy", "checkpoint"),
            hub_private_repo=self.config["output"].get("hub_private_repo", True),
        )

        # Load tokenizer and model (model load happens AFTER TrainingArguments
        # so ZeRO-3's zero.Init sharding, if configured, is already active)
        tokenizer = load_tokenizer(model_id)

        model = load_model_for_training(
            model_id,
            use_lora=use_lora,
            quantize=quantize,
            model_config=model_cfg,
        )

        # Apply PEFT if configured. `training.peft_method` selects the exact
        # variant (lora / dora / qlora / prefix_tuning / adapter) via the
        # shared factory in src/train/peft_factories.py — previously this was
        # hardcoded to plain LoRA regardless of `peft_method`, silently
        # making DoRA/QLoRA ablations byte-identical to the LoRA baseline.
        if use_lora:
            model = prepare_model_for_kbit_training(model)
            lora_cfg = self.config.get("lora", {})
            peft_method = self.train_cfg.get("peft_method", "lora")
            if peft_method not in ("lora", "dora", "qlora"):
                logger.warning(
                    f"training.peft_method={peft_method!r} is not lora/dora/qlora; "
                    "falling back to 'lora'."
                )
                peft_method = "lora"
            peft_config = create_peft_config(peft_method, lora_cfg)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # Load and prepare data
        logger.info("Loading and preparing training data...")
        aurora_loader = AuroraLoader(data_cfg)
        splits = aurora_loader.load_and_prepare()

        # Build mixture if configured
        mixture_name = self.config.get("data_mixture", "pt_only")
        if mixture_name != "pt_only":
            # NOTE: ReplayMixBuilder derives its shuffle seed from data_cfg's
            # dataset.seed (see src/data/replay_mix_builder.py) rather than
            # accepting an override here, so multi-seed CPT runs (see
            # scripts/run_multi_seed.sh) get identical replay sampling across
            # seeds — this is intentional: only the model's training seed
            # should vary, not which replay documents are eligible.
            mix_builder = ReplayMixBuilder(data_cfg)
            train_dataset = mix_builder.build_mixture(mixture_name, splits["train"])
        else:
            train_dataset = splits["train"]

        # Tokenize and pack. `packing.eos_separator`/`packing.mask_cross_doc_labels`
        # and `data.curriculum_sort` live in the TRAINING config (see e.g.
        # configs/train/ablation_packing.yaml), not the data config — reading
        # them from `data_cfg` (the old behavior) meant these ablation knobs
        # were always at their default value, making several ablation-matrix
        # variants silently identical to each other.
        max_seq_length = model_cfg["model"].get("max_seq_length", 8192)
        packing_cfg = self.config.get("packing", {})
        pack = packing_cfg.get("enabled", True)
        use_eos_separator = packing_cfg.get("eos_separator", True)
        mask_cross_doc_labels = packing_cfg.get("mask_cross_doc_labels", False)
        curriculum_sort = self.config.get("data", {}).get("curriculum_sort", False)

        train_tokenized = tokenize_for_cpt(
            train_dataset,
            tokenizer,
            max_seq_length=max_seq_length,
            pack=pack,
            curriculum_sort=curriculum_sort,
            mask_cross_doc_labels=mask_cross_doc_labels,
            use_eos_separator=use_eos_separator,
        )
        val_tokenized = tokenize_for_cpt(
            splits["validation"],
            tokenizer,
            max_seq_length=max_seq_length,
            pack=pack,
            use_eos_separator=use_eos_separator,
        )

        # Check for resume (auto-resume on Spot preemption recovery)
        resume_from = self.config.get("checkpointing", {}).get("resume_from_checkpoint")
        if resume_from is None:
            resume_from = find_latest_checkpoint(self.output_dir)
        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")

        # Callbacks
        metrics_logger = MetricsLogger(
            self.config.get("logging", {}).get("log_file", self.output_dir / "train_log.jsonl")
        )

        # Forgetting monitor: evaluate EN perplexity every N steps
        forgetting_cfg = self.config.get("forgetting_monitor", {})
        forgetting_monitor = ForgettingMonitorCallback(
            eval_interval_steps=forgetting_cfg.get("eval_interval_steps", eval_steps),
            max_eval_samples=forgetting_cfg.get("max_eval_samples", 100),
            metrics_logger=metrics_logger,
        )

        callbacks = [
            ThroughputCallback(metrics_logger, seq_length=max_seq_length),
            LocalMetricsCallback(metrics_logger),
            forgetting_monitor,
            PreemptionHandler(),
            GCSCheckpointSync(str(self.output_dir)),
            WandBCallback(
                project="gemma4-pt-br-adaptation",
                config=self.config,
                tags=[
                    self.config.get("experiment", {}).get("trilha", "unknown"),
                    "cpt",
                    "lora" if use_lora else "full",
                ],
            ),
        ]

        early_stopping_cfg = self.config.get("early_stopping", {})
        if early_stopping_cfg.get("enabled", False):
            from transformers import EarlyStoppingCallback

            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_cfg.get("patience", 5),
                    early_stopping_threshold=early_stopping_cfg.get("threshold", 0.001),
                )
            )

        # Data collator: preserve the labels pack_sequences already computed
        # (see PackedSequenceCollator docstring above) instead of
        # DataCollatorForLanguageModeling, which recomputes labels from
        # input_ids and would discard cross-document label masking.
        data_collator = PackedSequenceCollator()

        # Initialize trainer. `processing_class=tokenizer` is required for
        # ForgettingMonitorCallback (it reads the tokenizer via
        # `on_train_begin`'s `processing_class` kwarg) — without it, the
        # callback silently disabled itself on every run.
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        # Train
        logger.info("Starting continued pretraining...")
        train_result = trainer.train(resume_from_checkpoint=resume_from)

        # Save final model. Only the main process writes files — under
        # multi-GPU DDP/ZeRO-2 every rank holds identical full weights, so a
        # single writer is correct and avoids concurrent-write races.
        # KNOWN LIMITATION: saving a LoRA adapter under DeepSpeed ZeRO-3 needs
        # `deepspeed.zero.GatheredParameters` to reconstruct unsharded
        # adapter weights before `save_pretrained`; this repo's ZeRO-3 path
        # is intended for `use_lora=false` (full fine-tune, see
        # configs/train/cpt_main.yaml), where `trainer.save_model()` already
        # handles the gather correctly.
        final_dir = self.output_dir / "final"
        if trainer.is_world_process_zero():
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
        if trainer.is_world_process_zero():
            save_training_state(final_dir, state)

        logger.info(f"CPT completed in {elapsed:.1f}s. Model saved to {final_dir}")
        return state


def main():
    """CLI entry point for CPT."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Continued Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--override", nargs="*", help="Override config values (key=value)")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for multi-seed runs")
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides
    if args.override:
        import yaml as _yaml

        from src.utils.config_utils import merge_configs

        overrides = {}
        for o in args.override:
            key, value = o.split("=", 1)
            # Parse the value as YAML so `lora.r=64` becomes int 64, not the
            # string "64" (which would make `LoraConfig(r="64")` fail), and
            # `packing.mask_cross_doc_labels=true` becomes bool True, not the
            # truthy-but-wrong string "true".
            try:
                parsed_value = _yaml.safe_load(value)
            except _yaml.YAMLError:
                parsed_value = value
            keys = key.split(".")
            d = overrides
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = parsed_value
        config = merge_configs(config, overrides)

    # Seed override (for multi-seed experiments)
    if args.seed is not None:
        config.setdefault("experiment", {})["seed"] = args.seed
        # Append seed to output dir to avoid overwriting
        output_dir = config.get("output", {}).get("output_dir", "outputs/cpt")
        config["output"]["output_dir"] = f"{output_dir}_seed{args.seed}"
        logger.info(f"Multi-seed mode: seed={args.seed}, output={config['output']['output_dir']}")

    trainer = CPTTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
