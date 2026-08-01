"""Tests for DPOTrainerWrapper — no prior test coverage existed for this
file, which let a real bug ship silently: DPOConfig(max_prompt_length=...)
crashed with "unexpected keyword argument" on trl>=1.0 (that parameter was
removed; max_length is now the only length control). This exercises the
REAL DPOConfig/DPOTrainer construction call sites in dpo_trainer.py against
the installed trl version — model loading and actual training are mocked
out (no GPU/network needed), but the config-translation logic that broke
is not.
"""

from datasets import Dataset

from src.train.dpo_trainer import DPOTrainerWrapper


class _FakeDPOTrainer:
    """Stands in for trl.DPOTrainer — records the real DPOConfig it was
    constructed with (to prove that construction succeeded) without doing
    an actual forward/backward pass."""

    last_args = None
    last_kwargs = None

    def __init__(self, model, args, train_dataset, processing_class, callbacks):
        _FakeDPOTrainer.last_args = args
        _FakeDPOTrainer.last_kwargs = {
            "model": model,
            "train_dataset": train_dataset,
            "processing_class": processing_class,
            "callbacks": callbacks,
        }
        self.args = args

    def train(self, resume_from_checkpoint=None):
        class _Result:
            global_step = 5
            training_loss = 0.42

        return _Result()

    def is_world_process_zero(self):
        return False  # skip save_pretrained/save_model in the test

    def save_model(self, path):
        pass


def _base_config(tmp_path):
    return {
        "experiment": {"seed": 42},
        "model_config": {
            "model": {"base_id": "fake/base-model", "text_only_mode": True},
        },
        "training": {
            "use_lora": True,
            "num_train_epochs": 1,
            "max_steps": 10,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-6,
            "bf16": False,
        },
        "dpo": {
            "beta": 0.1,
            "loss_type": "sigmoid",
            "max_length": 512,
        },
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        },
        "checkpointing": {"save_steps": 500, "save_total_limit": 1},
        "logging": {"logging_steps": 10, "report_to": ["none"]},
        "output": {"output_dir": str(tmp_path / "dpo_out")},
    }


def test_run_constructs_real_dpo_config_without_crashing(tmp_path, monkeypatch):
    """The actual regression test: this used to raise
    `TypeError: DPOConfig.__init__() got an unexpected keyword argument
    'max_prompt_length'` on the installed trl version. If dpo_trainer.py
    regresses to passing that kwarg again, this test fails immediately
    instead of only at real-training time."""
    import src.train.dpo_trainer as dpo_module

    monkeypatch.setattr(dpo_module, "DPOTrainer", _FakeDPOTrainer)
    monkeypatch.setattr(dpo_module, "load_tokenizer", lambda model_id: object())
    monkeypatch.setattr(dpo_module, "load_model_for_training", lambda *a, **kw: object())
    monkeypatch.setattr(
        dpo_module,
        "get_peft_model",
        lambda model, peft_config: model,
    )
    monkeypatch.setattr(dpo_module, "prepare_model_for_kbit_training", lambda model: model)
    monkeypatch.setattr(dpo_module, "save_training_state", lambda *a, **kw: None)

    cfg = _base_config(tmp_path)
    wrapper = DPOTrainerWrapper(cfg)
    monkeypatch.setattr(
        wrapper,
        "_load_preference_data",
        lambda: Dataset.from_list(
            [{"prompt": "<|turn>user\nOi<turn|>\n", "chosen": "Olá!", "rejected": "..."}]
        ),
    )

    state = wrapper.run()

    assert _FakeDPOTrainer.last_args.max_length == 512
    assert _FakeDPOTrainer.last_args.beta == 0.1
    assert not hasattr(_FakeDPOTrainer.last_args, "max_prompt_length")
    assert state["global_step"] == 5
    assert state["training_loss"] == 0.42


def test_resume_from_checkpoint_auto_discovered(tmp_path, monkeypatch):
    """Previously dpo_trainer.py always called trainer.train() with no
    resume support at all — a DPO run interrupted mid-training silently
    restarted from scratch. Now it should auto-discover the latest
    checkpoint in output_dir, same as cpt_trainer.py/sft_trainer.py."""
    import src.train.dpo_trainer as dpo_module

    captured = {}

    class _RecordingFakeDPOTrainer(_FakeDPOTrainer):
        def train(self, resume_from_checkpoint=None):
            captured["resume_from_checkpoint"] = resume_from_checkpoint
            return super().train()

    monkeypatch.setattr(dpo_module, "DPOTrainer", _RecordingFakeDPOTrainer)
    monkeypatch.setattr(dpo_module, "load_tokenizer", lambda model_id: object())
    monkeypatch.setattr(dpo_module, "load_model_for_training", lambda *a, **kw: object())
    monkeypatch.setattr(dpo_module, "get_peft_model", lambda model, peft_config: model)
    monkeypatch.setattr(dpo_module, "prepare_model_for_kbit_training", lambda model: model)
    monkeypatch.setattr(dpo_module, "save_training_state", lambda *a, **kw: None)

    cfg = _base_config(tmp_path)
    output_dir = tmp_path / "dpo_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint-100"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer_state.json").write_text('{"global_step": 100}')

    wrapper = DPOTrainerWrapper(cfg)
    monkeypatch.setattr(
        wrapper,
        "_load_preference_data",
        lambda: Dataset.from_list([{"prompt": "p", "chosen": "c", "rejected": "r"}]),
    )

    wrapper.run()

    assert captured["resume_from_checkpoint"] == checkpoint_dir
