"""CLI entry point for gemma4-pt-br project.

Comandos disponíveis:
  preflight          — Valida ambiente (Python, pacotes, CUDA, disco, configs)
  data-validate      — Valida dados e manifesto de qualidade
  contamination-check — Checa contaminação dados × benchmarks
  tokenizer-audit    — Auditoria de fertilidade do tokenizer
  smoke              — Smoke test end-to-end em CPU
  train-cpt          — Continued Pretraining
  train-sft          — Supervised Fine-Tuning
  merge              — Residual Merge (task arithmetic)
  eval               — Avaliação em benchmarks
  report             — Gera relatórios e figuras
  run-all            — Pipeline completo
"""

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="gemma4pt",
    help="Gemma 4 Portuguese Adaptation — CLI de pesquisa",
    no_args_is_help=True,
)


# =============================================================================
# Opções globais comuns
# =============================================================================

DRY_RUN_HELP = "Simula execução sem efeitos colaterais"
TINY_HELP = "Usa dados/modelos mínimos para validação rápida"
CPU_ONLY_HELP = "Força execução em CPU (ignora GPU)"
RESUME_HELP = "Resume de checkpoint anterior"
NO_DOWNLOAD_HELP = "Não baixa modelos/dados do Hub"


# =============================================================================
# preflight
# =============================================================================


@app.command()
def preflight(
    check_gpu: bool = typer.Option(True, help="Verifica CUDA"),
    min_disk_gb: float = typer.Option(50.0, help="Espaço mínimo em disco (GB)"),
    strict: bool = typer.Option(False, help="Falha em warnings também"),
):
    """Valida ambiente antes de execução."""
    from src.preflight import run_preflight

    result = run_preflight(check_gpu=check_gpu, min_disk_gb=min_disk_gb)

    if strict and result.warnings:
        typer.echo("Modo strict: warnings tratados como falhas.")
        raise typer.Exit(1)
    if not result.passed:
        raise typer.Exit(1)


# =============================================================================
# data-validate
# =============================================================================


@app.command("data-validate")
def data_validate(
    config: str = typer.Option("configs/data/aurora_pt.yaml", help="Config de dados"),
    dry_run: bool = typer.Option(False, help=DRY_RUN_HELP),
):
    """Valida dados: manifesto de qualidade, splits, checksums."""
    from src.utils.config_utils import load_config

    cfg = load_config(config)
    typer.echo(f"Validando dados com config: {config}")

    if dry_run:
        typer.echo("[dry-run] Verificação estrutural apenas")
        typer.echo(f"  Corpus: {cfg.get('corpus', {}).get('name', 'N/A')}")
        typer.echo(f"  Splits configurados: {list(cfg.get('splits', {}).keys())}")
        typer.echo("[dry-run] OK — estrutura válida")
        return

    typer.echo("Carregando e validando datasets...")
    from src.data.aurora_loader import AuroraLoader

    loader = AuroraLoader(cfg)
    splits = loader.load_and_prepare()
    for split_name, ds in splits.items():
        typer.echo(f"  {split_name}: {len(ds)} exemplos")
    typer.echo("Validação concluída com sucesso.")


# =============================================================================
# contamination-check
# =============================================================================


@app.command("contamination-check")
def contamination_check(
    config: str = typer.Option("configs/eval/benchmarks.yaml", help="Config de eval"),
    corpus_config: str = typer.Option(
        "configs/data/aurora_pt.yaml", help="Config do corpus de treino a checar"
    ),
    sample_size: int = typer.Option(1000, help="Amostras do corpus para checar"),
    output_dir: str = typer.Option("outputs/contamination", help="Dir de saída"),
    dry_run: bool = typer.Option(False, help=DRY_RUN_HELP),
):
    """Checa contaminação entre corpus de treino e benchmarks."""
    from src.utils.config_utils import load_config

    if dry_run:
        typer.echo("[dry-run] Verificaria contaminação com:")
        typer.echo(f"  Config: {config}")
        typer.echo(f"  Amostras: {sample_size}")
        typer.echo(f"  Output: {output_dir}")
        return

    from src.data.aurora_loader import AuroraLoader
    from src.data.contamination_checks import run_contamination_report
    from src.eval.tasks import load_task

    eval_cfg = load_config(config)
    corpus_cfg = load_config(corpus_config)

    typer.echo(f"Amostrando {sample_size} documentos de {corpus_cfg['dataset']['hub_id']}...")
    loader = AuroraLoader(corpus_cfg)
    raw = loader.load_raw(streaming=True)
    corpus_sample = []
    for i, ex in enumerate(raw):
        if i >= sample_size:
            break
        text = ex.get("text", "")
        if text:
            corpus_sample.append(text)

    def _flatten_example_text(example: dict) -> str:
        """Join every string field of a benchmark example into one blob.

        Benchmark examples have heterogeneous schemas (question/alternatives
        for MC tasks, sentence/label for classification, etc.) — for
        contamination checking we just need the example's textual content,
        not any particular field, so concatenating every string value is a
        simple, task-agnostic way to get it.
        """
        parts = []
        for v in example.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                parts.extend(str(x) for x in v if isinstance(x, str))
        return " ".join(parts)

    benchmarks: dict[str, list[str]] = {}
    for name, bench_cfg in eval_cfg.get("benchmarks", {}).items():
        if not bench_cfg.get("enabled", True):
            continue
        try:
            # Dispatch by bench_cfg["task"] (the task implementation name,
            # e.g. "enem"), not by `name` (the benchmark config key, e.g.
            # "enem_2022") — several benchmark entries share one task.
            task = load_task(bench_cfg["task"])
            examples = task.load_data(bench_cfg)
        except Exception as e:  # noqa: BLE001 - best-effort, report and continue
            typer.echo(f"  [skip] {name}: falha ao carregar ({e})")
            continue
        if not examples:
            typer.echo(f"  [skip] {name}: 0 exemplos carregados")
            continue
        benchmarks[name] = [_flatten_example_text(ex) for ex in examples]

    typer.echo(
        f"Executando checagem de contaminação ({len(corpus_sample)} docs x "
        f"{len(benchmarks)} benchmarks)..."
    )
    report = run_contamination_report(corpus_sample, benchmarks, output_dir)
    typer.echo(f"Contamination check concluído. Resultados em: {output_dir}")
    for name, rate_by_method in report.get("summary", {}).items():
        exact_rate = rate_by_method.get("exact", 0.0)
        typer.echo(f"  {name}: exact_rate={exact_rate:.4f}")


# =============================================================================
# tokenizer-audit
# =============================================================================


@app.command("tokenizer-audit")
def tokenizer_audit(
    model_id: str = typer.Option("google/gemma-4-E4B", help="Model ID para tokenizer"),
    corpus_config: str = typer.Option(
        "configs/data/aurora_pt.yaml", help="Config do corpus PT-BR a amostrar"
    ),
    sample_size: int = typer.Option(5000, help="Documentos para amostrar"),
    output: str = typer.Option("outputs/tokenizer_audit.json", help="Path de saída do relatório"),
    dry_run: bool = typer.Option(False, help=DRY_RUN_HELP),
    no_download: bool = typer.Option(False, help=NO_DOWNLOAD_HELP),
):
    """Auditoria de fertilidade do tokenizer em corpus PT-BR."""
    if dry_run:
        typer.echo("[dry-run] Auditoria de tokenizer:")
        typer.echo(f"  Modelo: {model_id}")
        typer.echo(f"  Amostras: {sample_size}")
        return

    from src.data.aurora_loader import AuroraLoader
    from src.data.tokenizer_audit import run_tokenizer_audit
    from src.utils.config_utils import load_config
    from src.utils.hf_utils import load_tokenizer

    typer.echo(f"Carregando tokenizer de {model_id}...")
    tokenizer = load_tokenizer(model_id, local_files_only=no_download)

    typer.echo(f"Amostrando até {sample_size} documentos de {corpus_config}...")
    corpus_cfg = load_config(corpus_config)
    loader = AuroraLoader(corpus_cfg)
    raw = loader.load_raw(streaming=True)
    texts = []
    for i, ex in enumerate(raw):
        if i >= sample_size:
            break
        text = ex.get("text", "")
        if text:
            texts.append(text)

    from datasets import Dataset

    sample_ds = Dataset.from_dict({"text": texts})

    typer.echo(f"Executando auditoria de fertilidade em {len(texts)} documentos...")
    results = run_tokenizer_audit(tokenizer, sample_ds, sample_size=len(texts), output_path=output)
    typer.echo(f"Relatório salvo em: {output}")
    typer.echo(f"  Tokens/palavra: {results['tokens_per_word_mean']:.3f}")
    typer.echo(f"  Tokens/char: {results['tokens_per_char_mean']:.4f}")


# =============================================================================
# smoke
# =============================================================================


@app.command()
def smoke(
    cpu_only: bool = typer.Option(True, help=CPU_ONLY_HELP),
    verbose: bool = typer.Option(False, help="Output detalhado"),
):
    """Smoke test end-to-end em CPU com dados sintéticos."""
    typer.echo("=== Smoke Test End-to-End ===\n")

    # Delegate to the smoke test module
    from tests.smoke_test import run_smoke_test

    success = run_smoke_test(verbose=verbose)

    if success:
        typer.echo("\n[OK] Smoke test PASSED")
    else:
        typer.echo("\n[FAIL] Smoke test FAILED")
        raise typer.Exit(1)


# =============================================================================
# train-cpt
# =============================================================================


@app.command("train-cpt")
def train_cpt(
    config: str = typer.Argument(..., help="Path para config YAML de CPT"),
    dry_run: bool = typer.Option(False, help=DRY_RUN_HELP),
    tiny: bool = typer.Option(False, help=TINY_HELP),
    cpu_only: bool = typer.Option(False, help=CPU_ONLY_HELP),
    resume: bool = typer.Option(False, help=RESUME_HELP),
):
    """Executa Continued Pretraining."""
    from src.utils.config_utils import load_config, merge_configs

    cfg = load_config(config)

    if dry_run:
        typer.echo("[dry-run] CPT seria executado com:")
        typer.echo(f"  Config: {config}")
        typer.echo(f"  Model: {cfg.get('model_config', {}).get('model', {}).get('base_id', 'N/A')}")
        typer.echo(f"  Steps: {cfg.get('training', {}).get('max_steps', 'N/A')}")
        return

    if tiny:
        cfg = merge_configs(
            cfg,
            {
                "training": {"max_steps": 10, "per_device_train_batch_size": 1},
            },
        )
        typer.echo("[tiny] Usando max_steps=10, batch_size=1")

    if cpu_only:
        cfg = merge_configs(cfg, {"training": {"bf16": False, "tf32": False}})
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from src.train.cpt_trainer import CPTTrainer

    trainer = CPTTrainer(cfg)
    trainer.run()


# =============================================================================
# train-sft
# =============================================================================


@app.command("train-sft")
def train_sft(
    config: str = typer.Argument(..., help="Path para config YAML de SFT"),
    dry_run: bool = typer.Option(False, help=DRY_RUN_HELP),
    tiny: bool = typer.Option(False, help=TINY_HELP),
):
    """Executa Supervised Fine-Tuning."""
    from src.utils.config_utils import load_config, merge_configs

    cfg = load_config(config)

    if dry_run:
        typer.echo("[dry-run] SFT seria executado com config: " + config)
        return

    if tiny:
        cfg = merge_configs(cfg, {"training": {"max_steps": 10}})

    from src.train.sft_trainer import SFTTrainerWrapper

    trainer = SFTTrainerWrapper(cfg)
    trainer.run()


# =============================================================================
# merge
# =============================================================================


@app.command()
def merge(
    base_model: str = typer.Option(..., help="Base model ID"),
    instruct_model: str = typer.Option(..., help="Instruct model ID"),
    cpt_model: str = typer.Option(..., help="CPT model path"),
    alpha: list[float] = typer.Option([1.0], help="Alpha values para sweep"),
    output_dir: str = typer.Option("outputs/residual_merge", help="Diretório de saída"),
    dry_run: bool = typer.Option(False, help=DRY_RUN_HELP),
):
    """Executa Residual Merge (Task Arithmetic)."""
    if dry_run:
        typer.echo("[dry-run] Merge seria executado:")
        typer.echo(f"  Base: {base_model}")
        typer.echo(f"  Instruct: {instruct_model}")
        typer.echo(f"  CPT: {cpt_model}")
        typer.echo(f"  Alphas: {alpha}")
        return

    from src.train.residual_merge import alpha_sweep, compute_residual_merge

    if len(alpha) == 1:
        compute_residual_merge(base_model, instruct_model, cpt_model, alpha[0], output_dir)
    else:
        alpha_sweep(base_model, instruct_model, cpt_model, alpha, output_dir)


# =============================================================================
# eval
# =============================================================================


@app.command("eval")
def evaluate(
    config: str = typer.Option("configs/eval/benchmarks.yaml", help="Config de avaliação"),
    model: Optional[str] = typer.Option(None, help="Modelo específico para avaliar"),
    dry_run: bool = typer.Option(False, help=DRY_RUN_HELP),
    tiny: bool = typer.Option(False, help="Usa subset pequeno de cada benchmark"),
):
    """Executa avaliação em benchmarks."""
    if dry_run:
        from src.utils.config_utils import load_config

        cfg = load_config(config)
        benchmarks = cfg.get("benchmarks", {})
        typer.echo("[dry-run] Avaliação seria executada:")
        typer.echo(f"  Config: {config}")
        typer.echo(f"  Modelo: {model or 'todos do config'}")
        typer.echo(f"  Benchmarks: {len(benchmarks)} configurados")
        for name in list(benchmarks.keys())[:10]:
            typer.echo(f"    - {name}")
        return

    from src.eval.benchmark_runner import run_evaluation

    run_evaluation(config, model)


# =============================================================================
# report
# =============================================================================


@app.command()
def report(
    results_dir: str = typer.Option("reports", help="Diretório de resultados"),
):
    """Gera relatórios de avaliação (CSV, Markdown, figuras)."""
    from src.eval.report_builder import ReportBuilder, build_findings_for_paper

    results_path = Path(results_dir) / "eval_results.json"
    if not results_path.exists():
        typer.echo(f"Arquivo não encontrado: {results_path}")
        typer.echo("Execute 'gemma4pt eval' primeiro.")
        raise typer.Exit(1)

    with open(results_path) as f:
        results = json.load(f)
    builder = ReportBuilder(results, output_dir=results_dir)
    builder.build_all()
    build_findings_for_paper(results_dir)
    typer.echo(f"Relatórios gerados em {results_dir}/")


# =============================================================================
# run-all
# =============================================================================


@app.command("run-all")
def run_all(
    config: str = typer.Option("configs/train/cpt_pilot.yaml", help="Config de treino"),
    dry_run: bool = typer.Option(False, help=DRY_RUN_HELP),
    tiny: bool = typer.Option(False, help=TINY_HELP),
):
    """Pipeline completo: preflight → data → treino → merge → eval → report."""
    steps = [
        "preflight",
        "data-validate",
        "contamination-check",
        "train-cpt",
        "merge (se configurado)",
        "eval",
        "report",
    ]

    if dry_run:
        typer.echo("[dry-run] Pipeline completo seria executado:")
        for i, step in enumerate(steps, 1):
            typer.echo(f"  {i}. {step}")
        return

    typer.echo("=== Pipeline Completo ===\n")

    # Step 1: Preflight (CPU-safe)
    typer.echo("1/7 Preflight...")
    from src.preflight import run_preflight

    result = run_preflight(verbose=False)
    if not result.passed:
        typer.echo("Preflight falhou. Corrija os erros acima.")
        raise typer.Exit(1)
    typer.echo("  OK\n")

    # Step 2: Config/data structural validation (CPU-safe, no download)
    typer.echo("2/7 Validando estrutura de configs...")
    from src.utils.config_utils import load_config

    cfg = load_config(config)
    typer.echo(f"  Config carregado: {config}")
    typer.echo(f"  Modelo: {cfg.get('model_config', {}).get('model', {}).get('base_id', 'N/A')}")
    typer.echo("  OK\n")

    typer.echo(
        "Preflight e configs OK. As etapas restantes (data-validate real, "
        "contamination-check, train-cpt, merge, eval, report) fazem download "
        "de dados/modelos e/ou requerem GPU — execute-as individualmente:\n"
        f"  gemma4pt data-validate --config {cfg.get('data_config', 'configs/data/aurora_pt.yaml') if isinstance(cfg.get('data_config'), str) else 'configs/data/aurora_pt.yaml'}\n"
        f"  gemma4pt contamination-check\n"
        f"  gemma4pt train-cpt {config}\n"
        f"  gemma4pt merge --base-model ... --instruct-model ... --cpt-model outputs/.../final\n"
        f"  gemma4pt eval\n"
        f"  gemma4pt report"
    )


# =============================================================================
# manifest (gera manifesto de run)
# =============================================================================


@app.command()
def manifest(
    output: str = typer.Option("outputs/run_manifest.json", help="Path de saída"),
    config: Optional[str] = typer.Option(None, help="Config para incluir no manifesto"),
):
    """Gera manifesto de reprodutibilidade para a run atual."""
    import subprocess

    manifest_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": sys.version,
        "platform": {
            "system": sys.platform,
            "machine": __import__("platform").machine(),
            "node": __import__("platform").node(),
        },
        "packages": {},
        "git": {},
    }

    # Git info
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        manifest_data["git"]["sha"] = sha
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        manifest_data["git"]["dirty"] = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        manifest_data["git"]["sha"] = "unknown"

    # Package versions
    for pkg in ["torch", "transformers", "peft", "datasets", "accelerate", "trl"]:
        try:
            mod = importlib.import_module(pkg)
            manifest_data["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            manifest_data["packages"][pkg] = "not installed"

    # Config if provided
    if config:
        from src.utils.config_utils import load_config

        manifest_data["config"] = load_config(config)

    # Save
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest_data, f, indent=2, default=str)
    typer.echo(f"Manifesto salvo em: {out_path}")


if __name__ == "__main__":
    app()
