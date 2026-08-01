"""Real end-to-end CPT training mechanics — the one thing this repo had
never actually exercised.

Every other test in this suite either checks pure-Python logic or mocks the
model/trainer entirely. Nothing ever ran an actual forward+backward pass
through a real Gemma4ForConditionalGeneration, LoRA-wrapped, fed by the
real packing/collator pipeline, through a real HF `Trainer` — on GPU or
otherwise. There is no GPU available in this environment (or in this
repo's CI), so a full-scale run against the real ~2-30B-parameter Gemma 4
checkpoints isn't feasible here. This is the closest honest substitute:

- A genuinely tiny (~100K-parameter) `Gemma4ForConditionalGeneration`,
  randomly initialized locally (no download) from a config built with the
  real `Gemma4Config`/`Gemma4TextConfig` classes — so the actual production
  model code runs, just at a toy scale.
- A tiny local word-level tokenizer built by hand (no HF Hub call at all —
  this repo's CI deliberately runs with `HF_HUB_OFFLINE=1`/
  `TRANSFORMERS_OFFLINE=1`, see .github/workflows/ci.yml, specifically so
  no test ever depends on network availability; fetching even the small
  real Gemma 4 tokenizer here would violate that and break in a fresh CI
  runner with no HF cache).
- The REAL `tokenize_for_cpt`/packing pipeline (src/data/aurora_loader.py),
  the REAL `PackedSequenceCollator` (src/train/cpt_trainer.py), REAL LoRA
  wrapping via `create_peft_config`/`get_peft_model`, and a REAL
  `transformers.Trainer` — only the giant pretrained weights and the
  Aurora-PT corpus download are swapped out for tiny local equivalents.

This exercises the exact code paths CPTTrainerWrapper.run() would hit,
without needing to fight AuroraLoader's Hub dependency or GPU-only 4-bit
quantization to get there.
"""

import os

import pytest
import torch
from datasets import Dataset
from peft import get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.data.aurora_loader import tokenize_for_cpt
from src.train.cpt_trainer import PackedSequenceCollator
from src.train.peft_factories import create_peft_config
from src.utils.checkpointing import find_latest_checkpoint

_CORPUS = [
    "o gato dorme no sofa da sala",
    "a economia brasileira cresce todo ano",
    "o tribunal decidiu ontem sobre o caso",
    "os pesquisadores estudam o idioma portugues",
    "a equipe terminou o projeto no prazo",
    "o governo anunciou uma nova politica",
]


def _build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level tokenizer over the fixed toy corpus — fully local/offline,
    deterministic, no training step needed."""
    words = sorted({w for text in _CORPUS for w in text.split()})
    specials = ["<pad>", "<eos>", "<bos>", "<unk>"]
    vocab = {tok: i for i, tok in enumerate(specials + words)}

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )


def _build_tiny_model(
    vocab_size: int, tokenizer: PreTrainedTokenizerFast
) -> Gemma4ForConditionalGeneration:
    """A real Gemma4ForConditionalGeneration, randomly initialized at a toy
    scale (hidden_size=32, 2 layers) — same architecture class the
    production trainers load via AutoModelForCausalLM, just tiny."""
    text_cfg = Gemma4TextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=32,
        tie_word_embeddings=True,
        vocab_size_per_layer_input=vocab_size,
        hidden_size_per_layer_input=8,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    config = Gemma4Config(text_config=text_cfg)
    return Gemma4ForConditionalGeneration(config)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Builds the tiny model + tokenizer once per test module and saves
    them locally — each test reloads a fresh copy via
    AutoModelForCausalLM/AutoTokenizer (matching what load_model_for_training/
    load_tokenizer do in production) so training in one test can't leak
    state into another."""
    tokenizer = _build_tiny_tokenizer()
    model = _build_tiny_model(tokenizer.vocab_size, tokenizer)

    model_dir = tmp_path_factory.mktemp("tiny_gemma4") / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir


def _build_trainer(model_dir, output_dir, max_steps, save_steps=2) -> Trainer:
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    peft_config = create_peft_config(
        "lora", {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}
    )
    lora_model = get_peft_model(model, peft_config)

    dataset = Dataset.from_list([{"text": t} for t in _CORPUS * 20])
    tokenized = tokenize_for_cpt(dataset, tokenizer, max_seq_length=16, pack=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        logging_steps=100,
        save_steps=save_steps,
        save_total_limit=5,
        report_to=[],
        bf16=False,
        fp16=False,
        disable_tqdm=True,
        seed=42,
        data_seed=42,
    )
    return Trainer(
        model=lora_model,
        args=args,
        train_dataset=tokenized,
        data_collator=PackedSequenceCollator(),
    )


def test_tiny_model_trains_with_finite_loss(tiny_model_dir, tmp_path):
    """The core P0 gap this fills: prove a real forward+backward+optimizer
    step through the actual production code path (Gemma4ForConditionalGeneration
    + LoRA + tokenize_for_cpt packing + PackedSequenceCollator + Trainer)
    runs to completion on CPU and produces a finite loss — not mocked,
    not a dry-run, an actual training step."""
    output_dir = tmp_path / "out"
    trainer = _build_trainer(tiny_model_dir, output_dir, max_steps=4)

    result = trainer.train()

    assert trainer.state.global_step == 4
    assert result.training_loss == result.training_loss  # not NaN
    assert result.training_loss > 0
    assert (output_dir / "checkpoint-2").exists()
    assert (output_dir / "checkpoint-4").exists()


def test_interrupted_training_resumes_from_correct_step(tiny_model_dir, tmp_path):
    """Simulates a genuine crash-and-resume: train() only up to step 2 (as
    if the process died there), then build a COMPLETELY FRESH Trainer
    instance (fresh model reload, freshly random-initialized LoRA adapters
    — simulating a new process) pointed at max_steps=4, and confirm it
    auto-discovers the checkpoint and resumes from step 2 rather than
    silently restarting from scratch. This is the exact pattern
    cpt_trainer.py/sft_trainer.py use for real Spot-preemption recovery,
    but it had never actually been exercised end-to-end before.

    global_step reaching 4 alone doesn't prove resume worked (a silent
    restart-from-scratch would also reach 4). The rigorous check: the
    fresh trainer's LoRA weights are random noise before training starts
    (get_peft_model() always reinitializes them) — capture them via an
    on_train_begin callback, which fires AFTER Trainer internally loads
    resume_from_checkpoint but BEFORE any further optimizer step. If
    resume genuinely worked, those weights must exactly equal what was
    saved to checkpoint-2, not the pre-resume random init.
    """
    output_dir = tmp_path / "out"

    interrupted_trainer = _build_trainer(tiny_model_dir, output_dir, max_steps=2)
    interrupted_trainer.train()
    assert interrupted_trainer.state.global_step == 2
    assert sorted(os.listdir(output_dir)) == ["checkpoint-2"]
    checkpoint_2_weights = {
        n: p.detach().clone()
        for n, p in interrupted_trainer.model.named_parameters()
        if "lora" in n
    }

    resume_from = find_latest_checkpoint(output_dir)
    assert resume_from is not None and resume_from.name == "checkpoint-2"

    resumed_trainer = _build_trainer(tiny_model_dir, output_dir, max_steps=4)
    fresh_random_weights = {
        n: p.detach().clone() for n, p in resumed_trainer.model.named_parameters() if "lora" in n
    }
    loaded_at_train_begin = {}

    class _CaptureAtTrainBegin(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            loaded_at_train_begin.update(
                {n: p.detach().clone() for n, p in model.named_parameters() if "lora" in n}
            )

    resumed_trainer.add_callback(_CaptureAtTrainBegin())
    resumed_trainer.train(resume_from_checkpoint=str(resume_from))

    assert resumed_trainer.state.global_step == 4
    assert (output_dir / "checkpoint-4").exists()

    # The fresh trainer's pre-resume init must NOT already match
    # checkpoint-2 (otherwise this test would trivially pass for the wrong
    # reason — confirms the two LoRA inits are genuinely different draws).
    assert any(
        not torch.equal(fresh_random_weights[n], checkpoint_2_weights[n])
        for n in checkpoint_2_weights
    )
    # But by on_train_begin, the checkpoint's weights must have been loaded
    # exactly — proving this was a real resume, not a silent restart.
    for name, expected in checkpoint_2_weights.items():
        assert torch.equal(loaded_at_train_begin[name], expected), (
            f"{name} at resume's on_train_begin doesn't match checkpoint-2 — "
            "resume_from_checkpoint did not actually restore model state"
        )
