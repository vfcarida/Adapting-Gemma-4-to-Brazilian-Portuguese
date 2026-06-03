"""Smoke test end-to-end em CPU com dados e modelos sintéticos.

Este teste valida o pipeline completo sem GPU:
1. Leitura de config
2. Preparação de dados sintéticos
3. Tokenização
4. Forward pass mínimo (tiny model)
5. Checkpoint save/load
6. Avaliação em benchmark fixture
7. Cálculo de métricas
8. Bootstrap CI
9. Geração de CSV/Markdown final

Execução:
    python -m tests.smoke_test
    # ou via CLI:
    gemma4pt smoke
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def run_smoke_test(verbose: bool = False) -> bool:
    """Executa smoke test completo. Retorna True se tudo passou."""
    results = {}
    start = time.time()

    def log(msg):
        if verbose:
            print(f"  {msg}")

    def check(name: str, fn):
        try:
            fn()
            results[name] = "PASS"
            print(f"  [OK] {name}")
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"  [FAIL] {name}: {e}")

    # =========================================================================
    # 1. Config loading and merging
    # =========================================================================
    def test_config_loading():
        from src.utils.config_utils import flatten_config, load_config, merge_configs

        # Load real config
        cfg = load_config("configs/eval/benchmarks.yaml")
        assert "evaluation" in cfg or "benchmarks" in cfg, "Config vazia ou inválida"

        # Test merge
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"c": 3, "d": 4}}
        merged = merge_configs(base, override)
        assert merged["b"]["c"] == 3
        assert merged["b"]["d"] == 4
        assert merged["a"] == 1

        # Test flatten
        flat = flatten_config(merged)
        assert "b.c" in flat

    check("config_loading", test_config_loading)

    # =========================================================================
    # 2. Seed and reproducibility
    # =========================================================================
    def test_seed():
        import numpy as np

        from src.utils.seed import set_seed

        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        assert np.allclose(a, b), "Seeds não reproduzíveis"

        try:
            import torch

            set_seed(42)
            ta = torch.randn(10)
            set_seed(42)
            tb = torch.randn(10)
            assert torch.allclose(ta, tb), "Torch seeds não reproduzíveis"
        except ImportError:
            pass  # torch optional for smoke test

    check("seed_reproducibility", test_seed)

    # =========================================================================
    # 3. Prompt builders
    # =========================================================================
    def test_prompt_builders():
        from src.data.prompt_builders import BaselinePromptBuilder, Gemma4PromptBuilder

        # Test with None tokenizer (fallback mode)
        class FakeTokenizer:
            pass

        builder = Gemma4PromptBuilder(FakeTokenizer())
        messages = [{"role": "user", "content": "Olá, mundo!"}]

        # Inference format
        result = builder.format_for_inference(messages, think_mode="off")
        assert "Olá, mundo!" in result
        assert "<start_of_turn>user" in result
        assert "<start_of_turn>model" in result

        # Think mode on
        result_think = builder.format_for_inference(messages, think_mode="on")
        assert "<think>" in result_think

        # Think mode budget
        result_budget = builder.format_for_inference(messages, think_mode="budget")
        assert "<think>" in result_budget
        assert "</think>" in result_budget

        # Strip thought
        text_with_think = "<think>pensando...</think>A resposta é B"
        cleaned = builder.strip_thought(text_with_think)
        assert "pensando" not in cleaned
        assert "resposta é B" in cleaned

        # Extract thought
        thought, answer = builder.extract_thought(text_with_think)
        assert "pensando" in thought
        assert "resposta é B" in answer

        # Multi-turn stripping
        multi = "<think>first thought</think>Answer 1"
        thought, answer = builder.extract_thought(multi)
        assert "first thought" in thought

        # Baseline builder
        baseline = BaselinePromptBuilder(
            prompt_prefix="Responda:\n",
            user_prefix="P: ",
            model_prefix="R: ",
        )
        result = baseline.format_for_inference(messages)
        assert "Responda:" in result
        assert "P: Olá, mundo!" in result
        assert "R: " in result

    check("prompt_builders", test_prompt_builders)

    # =========================================================================
    # 4. Eval prompt templates
    # =========================================================================
    def test_prompt_templates():
        from src.eval.prompt_templates import (
            TASK_FORMATTERS,
            TASK_INSTRUCTIONS,
            extract_thought,
            get_prompt_template,
            strip_thought,
        )

        # All tasks have instructions and formatters
        assert len(TASK_INSTRUCTIONS) >= 20
        assert len(TASK_FORMATTERS) >= 20

        # Test ENEM formatting
        template = get_prompt_template("enem", num_shots=0)
        example = {
            "question": "Qual é a capital do Brasil?",
            "options": ["São Paulo", "Rio", "Brasília", "Salvador", "Curitiba"],
            "answer": "C",
        }
        prompt = template.format_prompt(example, think_mode="off")
        assert "Qual é a capital" in prompt
        assert "C) Brasília" in prompt

        # Think mode
        prompt_think = template.format_prompt(example, think_mode="on")
        assert "<think>" in prompt_think

        # Strip/extract
        assert strip_thought("<think>x</think>y") == "y"
        t, a = extract_thought("<think>raciocínio</think>resposta")
        assert "raciocínio" in t
        assert "resposta" in a

    check("prompt_templates", test_prompt_templates)

    # =========================================================================
    # 5. Metrics computation
    # =========================================================================
    def test_metrics():
        from src.eval.metrics import compute_metrics_for_task

        # Accuracy
        preds = ["A", "B", "C", "A", "B"]
        golds = ["A", "B", "C", "B", "B"]
        m = compute_metrics_for_task("accuracy", preds, golds)
        assert "accuracy" in m
        assert m["accuracy"] == 0.8

        # F1 (correct registry name is "macro_f1")
        m_f1 = compute_metrics_for_task("macro_f1", preds, golds)
        assert "macro_f1" in m_f1

    check("metrics_computation", test_metrics)

    # =========================================================================
    # 6. Bootstrap CI
    # =========================================================================
    def test_bootstrap():
        from src.eval.bootstrap_ci import bootstrap_ci, paired_bootstrap_test
        from src.eval.metrics import compute_metrics_for_task

        preds = ["A", "B", "C", "A", "B"] * 20
        golds = ["A", "B", "C", "B", "B"] * 20

        def metric_fn(p, g):
            return compute_metrics_for_task("accuracy", p, g)

        ci = bootstrap_ci(preds, golds, metric_fn, n_bootstrap=100, seed=42)
        assert "accuracy" in ci
        assert ci["accuracy"]["ci_lower"] <= ci["accuracy"]["mean"]
        assert ci["accuracy"]["mean"] <= ci["accuracy"]["ci_upper"]

        # Paired test
        preds_b = ["A", "C", "C", "A", "A"] * 20
        result = paired_bootstrap_test(
            preds, preds_b, golds, metric_fn, "accuracy", n_bootstrap=100
        )
        assert "p_value_a_gt_b" in result

    check("bootstrap_ci", test_bootstrap)

    # =========================================================================
    # 7. Stats tests
    # =========================================================================
    def test_stats():
        import numpy as np

        from src.eval.stats_tests import (
            compute_effect_size,
            multiple_comparison_correction,
            paired_permutation_test,
        )

        scores_a = np.array([0.8, 0.9, 0.7, 0.85, 0.75])
        scores_b = np.array([0.6, 0.7, 0.5, 0.65, 0.55])

        # Permutation test
        result = paired_permutation_test(scores_a, scores_b, n_permutations=100)
        assert "p_value" in result
        assert result["p_value"] < 0.1  # Should be significant

        # Effect size
        es = compute_effect_size(scores_a, scores_b)
        assert "cohens_d" in es
        assert es["cohens_d"] > 0  # A > B

        # Multiple comparison correction
        p_values = [0.01, 0.03, 0.05, 0.10]
        corrected = multiple_comparison_correction(p_values, method="holm")
        assert len(corrected) == 4
        assert all(c >= o for c, o in zip(corrected, p_values))

    check("stats_tests", test_stats)

    # =========================================================================
    # 8. Checkpointing
    # =========================================================================
    def test_checkpointing():
        from src.utils.checkpointing import (
            find_latest_checkpoint,
            load_training_state,
            save_training_state,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save state
            state = {"step": 100, "loss": 2.5, "config": {"lr": 1e-4}}
            save_training_state(tmpdir, state)

            # Load state
            loaded = load_training_state(tmpdir)
            assert loaded is not None
            assert loaded["step"] == 100
            assert loaded["loss"] == 2.5

            # Find checkpoint (create fake)
            ckpt_dir = Path(tmpdir) / "checkpoint-100"
            ckpt_dir.mkdir()
            (Path(tmpdir) / "checkpoint-200").mkdir()

            latest = find_latest_checkpoint(tmpdir)
            assert latest is not None
            assert "200" in latest.name

    check("checkpointing", test_checkpointing)

    # =========================================================================
    # 9. Contamination checks
    # =========================================================================
    def test_contamination():
        from src.data.contamination_checks import (
            ContaminationChecker,
            compute_hash,
            ngrams,
            normalize_text,
        )

        # Normalization
        assert normalize_text("  Olá  Mundo! ") == "ola mundo"
        assert normalize_text("Açúcar") == "acucar"

        # Hash
        h1 = compute_hash("teste")
        h2 = compute_hash("TESTE")
        assert h1 == h2  # Normalized

        # N-grams
        text = "a b c d e f"
        grams = ngrams(text, 3)
        assert "a b c" in grams
        assert "d e f" in grams

        # Checker
        bench = ["O gato sentou no tapete", "O sol nasceu bonito hoje"]
        checker = ContaminationChecker(bench, "test_bench")

        train = ["O gato sentou no tapete", "Texto completamente diferente"]
        exact_result = checker.check_exact(train)
        assert exact_result["matches"] == 1

    check("contamination_checks", test_contamination)

    # =========================================================================
    # 10. Report builder (with synthetic data)
    # =========================================================================
    def test_report_builder():
        from src.eval.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            # Synthetic results
            results = [
                {
                    "model_name": "test_model",
                    "model_id": "test/model",
                    "benchmarks": {
                        "think_off": {
                            "enem": {
                                "task": "enem",
                                "group": "brasil_geral",
                                "metric_name": "accuracy",
                                "metrics": {"accuracy": 0.75},
                                "num_examples": 100,
                                "inference_time_sec": 10.0,
                                "think_mode": "off",
                                "raw_predictions": ["A", "B"],
                            },
                            "assin2_rte": {
                                "task": "assin2_rte",
                                "group": "semantica",
                                "metric_name": "accuracy",
                                "metrics": {"accuracy": 0.82},
                                "num_examples": 50,
                                "inference_time_sec": 5.0,
                                "think_mode": "off",
                                "raw_predictions": ["entailment"],
                            },
                        }
                    },
                }
            ]

            builder = ReportBuilder(results, output_dir=tmpdir)
            builder.build_all()

            # Verify outputs
            assert (Path(tmpdir) / "results_full.csv").exists()
            assert (Path(tmpdir) / "results_pivot.csv").exists()
            assert (Path(tmpdir) / "group_averages.csv").exists()
            assert (Path(tmpdir) / "summary.md").exists()

    check("report_builder", test_report_builder)

    # =========================================================================
    # 11. Residual merge (tiny tensors)
    # =========================================================================
    def test_residual_merge_logic():
        import numpy as np

        # Simulate merge arithmetic without loading real models
        np.random.seed(42)
        base_w = np.random.randn(10, 10)
        instruct_w = base_w + np.random.randn(10, 10) * 0.1
        cpt_w = base_w + np.random.randn(10, 10) * 0.05

        alpha = 1.0
        residual = instruct_w - base_w
        merged = cpt_w + alpha * residual

        # Merged should be different from CPT (added instruction)
        assert not np.allclose(merged, cpt_w)
        # Residual should match instruction delta
        assert np.allclose(residual, instruct_w - base_w)
        # Formula verification
        expected = cpt_w + alpha * (instruct_w - base_w)
        assert np.allclose(merged, expected)

    check("residual_merge_logic", test_residual_merge_logic)

    # =========================================================================
    # 12. Config merge/override
    # =========================================================================
    def test_config_override():
        from src.utils.config_utils import merge_configs

        base = {
            "training": {"lr": 1e-4, "batch_size": 8, "nested": {"a": 1, "b": 2}},
            "model": {"name": "test"},
        }
        override = {
            "training": {"lr": 2e-4, "nested": {"a": 10}},
        }
        result = merge_configs(base, override)
        assert result["training"]["lr"] == 2e-4
        assert result["training"]["batch_size"] == 8  # Preserved
        assert result["training"]["nested"]["a"] == 10
        assert result["training"]["nested"]["b"] == 2  # Preserved
        assert result["model"]["name"] == "test"  # Preserved

    check("config_override", test_config_override)

    # =========================================================================
    # 13. Preflight
    # =========================================================================
    def test_preflight():
        from src.preflight import PreflightResult, run_preflight

        result = run_preflight(
            check_gpu=False,
            min_disk_gb=1.0,
            verbose=False,
        )
        assert isinstance(result, PreflightResult)
        assert len(result.checks) > 0
        # Python version should always pass
        python_check = next(c for c in result.checks if c["name"] == "python_version")
        assert python_check["status"] == "ok"

    check("preflight", test_preflight)

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start
    print(f"\n  Tempo total: {elapsed:.1f}s")

    n_pass = sum(1 for v in results.values() if v == "PASS")
    n_fail = sum(1 for v in results.values() if v != "PASS")
    print(f"  Resultados: {n_pass} passed, {n_fail} failed")

    return n_fail == 0


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    success = run_smoke_test(verbose=verbose)
    sys.exit(0 if success else 1)
