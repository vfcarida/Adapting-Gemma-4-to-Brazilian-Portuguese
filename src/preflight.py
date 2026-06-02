"""Preflight checks — validação de ambiente antes de execução.

Verifica dependências, versões, autenticação, espaço em disco,
CUDA, paths de configs e arquivos obrigatórios.
"""

import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Versões mínimas testadas para compatibilidade com Gemma 4
REQUIRED_PACKAGES = {
    "torch": "2.2.0",
    "transformers": "4.45.0",
    "datasets": "3.0.0",
    "accelerate": "1.0.0",
    "peft": "0.13.0",
    "trl": "0.12.0",
    "numpy": "1.26.0",
    "scipy": "1.12.0",
    "pyyaml": "6.0.1",
}

OPTIONAL_PACKAGES = {
    "vllm": "0.6.0",
    "deepspeed": "0.14.0",
    "bitsandbytes": "0.44.0",
    "bert_score": "0.3.13",
}

REQUIRED_FILES = [
    "pyproject.toml",
    "configs/eval/benchmarks.yaml",
    "configs/model/gemma4_e4b.yaml",
    "configs/train/cpt_pilot.yaml",
    "src/eval/benchmark_runner.py",
    "src/train/cpt_trainer.py",
]


class PreflightResult:
    """Resultado de uma verificação preflight."""

    def __init__(self):
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, status: str, message: str, details: str = ""):
        """Adiciona resultado de check. status: 'ok', 'warn', 'fail'."""
        self.checks.append({
            "name": name,
            "status": status,
            "message": message,
            "details": details,
        })

    @property
    def passed(self) -> bool:
        return not any(c["status"] == "fail" for c in self.checks)

    @property
    def warnings(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "warn"]

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "fail"]

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            icon = {"ok": "[OK]", "warn": "[WARN]", "fail": "[FAIL]"}[c["status"]]
            lines.append(f"  {icon} {c['name']}: {c['message']}")
            if c["details"]:
                lines.append(f"        {c['details']}")
        n_ok = sum(1 for c in self.checks if c["status"] == "ok")
        n_warn = len(self.warnings)
        n_fail = len(self.failures)
        lines.append(f"\n  Total: {n_ok} ok, {n_warn} warnings, {n_fail} failures")
        if self.passed:
            lines.append("  STATUS: READY")
        else:
            lines.append("  STATUS: NOT READY — fix failures above")
        return "\n".join(lines)


def _parse_version(v: str) -> tuple:
    """Parse version string to tuple for comparison."""
    parts = []
    for p in v.split(".")[:3]:
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def check_python_version(result: PreflightResult) -> None:
    """Verifica Python >= 3.10."""
    v = sys.version_info
    if v >= (3, 10):
        result.add("python_version", "ok", f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        result.add(
            "python_version", "fail",
            f"Python {v.major}.{v.minor}.{v.micro} — requer >= 3.10",
            "Instale Python 3.10+ via pyenv ou conda",
        )


def check_required_packages(result: PreflightResult) -> None:
    """Verifica pacotes obrigatórios e versões mínimas."""
    for pkg, min_ver in REQUIRED_PACKAGES.items():
        try:
            module_name = "yaml" if pkg == "pyyaml" else pkg
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "0.0.0")
            if _parse_version(version) >= _parse_version(min_ver):
                result.add(f"pkg_{pkg}", "ok", f"{pkg}=={version}")
            else:
                result.add(
                    f"pkg_{pkg}", "fail",
                    f"{pkg}=={version} < {min_ver}",
                    f"pip install '{pkg}>={min_ver}'",
                )
        except ImportError:
            result.add(
                f"pkg_{pkg}", "fail",
                f"{pkg} não instalado",
                f"pip install '{pkg}>={min_ver}'",
            )


def check_optional_packages(result: PreflightResult) -> None:
    """Verifica pacotes opcionais (apenas warnings)."""
    for pkg, min_ver in OPTIONAL_PACKAGES.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "0.0.0")
            result.add(f"opt_{pkg}", "ok", f"{pkg}=={version} (opcional)")
        except ImportError:
            result.add(
                f"opt_{pkg}", "warn",
                f"{pkg} não instalado (opcional)",
                f"pip install '{pkg}>={min_ver}' para funcionalidades extras",
            )


def check_cuda(result: PreflightResult) -> None:
    """Verifica disponibilidade de CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            result.add(
                "cuda", "ok",
                f"CUDA disponível: {device_count} GPU(s) — {device_name} ({mem:.1f}GB)",
            )
        else:
            result.add(
                "cuda", "warn",
                "CUDA não disponível — apenas CPU",
                "Treinamento requer GPU. Smoke tests e avaliação local rodam em CPU.",
            )
    except ImportError:
        result.add("cuda", "warn", "torch não instalado — não foi possível verificar CUDA")


def check_disk_space(result: PreflightResult, min_gb: float = 50.0) -> None:
    """Verifica espaço em disco disponível."""
    usage = shutil.disk_usage(".")
    free_gb = usage.free / (1024**3)
    if free_gb >= min_gb:
        result.add("disk_space", "ok", f"{free_gb:.1f}GB livres")
    elif free_gb >= 10.0:
        result.add(
            "disk_space", "warn",
            f"{free_gb:.1f}GB livres (recomendado: {min_gb}GB para modelos grandes)",
        )
    else:
        result.add(
            "disk_space", "fail",
            f"Apenas {free_gb:.1f}GB livres — insuficiente",
            "Libere espaço ou use storage externo para checkpoints.",
        )


def check_hf_auth(result: PreflightResult) -> None:
    """Verifica autenticação HuggingFace."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        result.add("hf_auth", "ok", "HF_TOKEN encontrado no ambiente")
        return

    # Check HuggingFace cache
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if hf_token_path.exists():
        result.add("hf_auth", "ok", "Token HF encontrado em ~/.cache/huggingface/token")
    else:
        result.add(
            "hf_auth", "warn",
            "Sem autenticação HF detectada",
            "Execute 'huggingface-cli login' ou defina HF_TOKEN para acessar modelos gated.",
        )


def check_required_files(result: PreflightResult, project_root: Path | None = None) -> None:
    """Verifica presença de arquivos obrigatórios do projeto."""
    root = project_root or Path(".")
    missing = []
    for f in REQUIRED_FILES:
        if not (root / f).exists():
            missing.append(f)

    if not missing:
        result.add("required_files", "ok", f"Todos os {len(REQUIRED_FILES)} arquivos obrigatórios presentes")
    else:
        result.add(
            "required_files", "fail",
            f"{len(missing)} arquivo(s) obrigatório(s) ausente(s)",
            f"Ausentes: {', '.join(missing[:5])}",
        )


def check_write_permissions(result: PreflightResult) -> None:
    """Verifica permissões de escrita em diretórios de saída."""
    dirs_to_check = ["outputs", "reports", "."]
    for d in dirs_to_check:
        path = Path(d)
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except PermissionError:
            result.add(
                "write_permissions", "fail",
                f"Sem permissão de escrita em '{d}'",
                f"chmod +w {d}",
            )
            return
    result.add("write_permissions", "ok", "Permissões de escrita OK")


def check_configs_valid(result: PreflightResult, project_root: Path | None = None) -> None:
    """Valida que configs YAML são parseable e coerentes."""
    import yaml
    root = project_root or Path(".")
    config_dir = root / "configs"

    if not config_dir.exists():
        result.add("configs_valid", "fail", "Diretório configs/ não encontrado")
        return

    errors = []
    count = 0
    for yaml_file in config_dir.rglob("*.yaml"):
        count += 1
        try:
            with open(yaml_file) as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            errors.append(f"{yaml_file.relative_to(root)}: {e}")

    if errors:
        result.add(
            "configs_valid", "fail",
            f"{len(errors)} config(s) com erro de parse",
            "; ".join(errors[:3]),
        )
    else:
        result.add("configs_valid", "ok", f"{count} configs YAML válidos")


def run_preflight(
    project_root: Path | None = None,
    check_gpu: bool = True,
    min_disk_gb: float = 50.0,
    verbose: bool = True,
) -> PreflightResult:
    """Executa todas as verificações preflight.

    Args:
        project_root: Raiz do projeto (default: cwd).
        check_gpu: Se deve verificar CUDA.
        min_disk_gb: Espaço mínimo em disco (GB).
        verbose: Se deve imprimir resultado.

    Returns:
        PreflightResult com todos os checks.
    """
    result = PreflightResult()

    check_python_version(result)
    check_required_packages(result)
    check_optional_packages(result)
    if check_gpu:
        check_cuda(result)
    check_disk_space(result, min_disk_gb)
    check_hf_auth(result)
    check_required_files(result, project_root)
    check_write_permissions(result)
    check_configs_valid(result, project_root)

    if verbose:
        print("\n=== Preflight Check ===\n")
        print(result.summary())
        print()

    return result


if __name__ == "__main__":
    result = run_preflight()
    sys.exit(0 if result.passed else 1)
