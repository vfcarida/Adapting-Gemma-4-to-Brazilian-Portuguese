#!/bin/bash
# =============================================================================
# Preflight Check — Validate environment before launching training on GCP
# =============================================================================
# Checks GPU availability, VRAM, disk space, HF token, GCS access, and
# Python dependencies. Run this BEFORE launching expensive training jobs.
#
# Usage:
#   bash scripts/preflight.sh                    # Full check
#   bash scripts/preflight.sh --config configs/train/cpt_pilot.yaml  # With VRAM estimate
#
# Exit codes:
#   0 = All checks passed
#   1 = Critical failure (will not train successfully)
#   2 = Warning (may work but suboptimal)
# =============================================================================

set -uo pipefail

CONFIG=""
WARNINGS=0
ERRORS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Gemma 4 PT-BR — Preflight Check                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# --- 1. Python Environment ---
echo "─── Python Environment ───"
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "  ✓ ${PYTHON_VERSION}"
else
    echo "  ✗ python3 not found"
    ERRORS=$((ERRORS + 1))
fi

# Check critical packages
for pkg in torch transformers datasets peft accelerate; do
    if python3 -c "import ${pkg}" 2>/dev/null; then
        VERSION=$(python3 -c "import ${pkg}; print(${pkg}.__version__)" 2>/dev/null || echo "?")
        echo "  ✓ ${pkg}==${VERSION}"
    else
        echo "  ✗ ${pkg} not installed"
        ERRORS=$((ERRORS + 1))
    fi
done

# Optional packages
for pkg in vllm wandb deepspeed bitsandbytes; do
    if python3 -c "import ${pkg}" 2>/dev/null; then
        VERSION=$(python3 -c "import ${pkg}; print(${pkg}.__version__)" 2>/dev/null || echo "?")
        echo "  ○ ${pkg}==${VERSION} (optional)"
    else
        echo "  ○ ${pkg} not installed (optional)"
    fi
done
echo ""

# --- 2. GPU Check ---
echo "─── GPU Availability ───"
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_VRAM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')")
    echo "  ✓ ${GPU_COUNT}x ${GPU_NAME} (${GPU_VRAM} each)"

    # VRAM estimation if config provided
    if [ -n "${CONFIG}" ]; then
        echo ""
        echo "  VRAM Estimate (from config):"
        python3 -c "
from src.utils.hf_utils import estimate_vram_gb
from src.utils.config_utils import load_config
import json

config = load_config('${CONFIG}')
model_cfg_path = config.get('model_config', '')
if model_cfg_path:
    model_cfg = load_config(model_cfg_path)
else:
    model_cfg = {}

# Extract parameters from config
params_total = model_cfg.get('model', {}).get('params_total', '4B')
params_b = float(params_total.replace('B', '')) if isinstance(params_total, str) else float(params_total)
seq_length = model_cfg.get('model', {}).get('max_seq_length', 8192)
batch_size = config.get('training', {}).get('per_device_train_batch_size', 1)
use_lora = config.get('training', {}).get('use_lora', False)
lora_r = config.get('lora', {}).get('r', 64)
gc = config.get('training', {}).get('gradient_checkpointing', True)

est = estimate_vram_gb(
    model_params_b=params_b,
    seq_length=seq_length,
    batch_size=batch_size,
    use_lora=use_lora,
    lora_r=lora_r,
    gradient_checkpointing=gc,
)

print(f'    Model weights:    {est[\"model_weights_gb\"]:>6.1f} GB')
print(f'    Optimizer states: {est[\"optimizer_states_gb\"]:>6.1f} GB')
print(f'    Gradients:        {est[\"gradients_gb\"]:>6.1f} GB')
print(f'    Activations:      {est[\"activations_gb\"]:>6.1f} GB')
print(f'    Overhead:         {est[\"overhead_gb\"]:>6.1f} GB')
print(f'    ─────────────────────────────')
print(f'    TOTAL:            {est[\"total_estimated_gb\"]:>6.1f} GB')
print(f'    Recommended:      {est[\"recommended_gpu\"]}')
" 2>/dev/null || echo "    (Could not compute VRAM estimate)"
    fi
else
    echo "  ✗ No GPU available (CUDA not detected)"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# --- 3. Disk Space ---
echo "─── Disk Space ───"
DISK_AVAIL=$(df -BG . 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
if [ -z "${DISK_AVAIL}" ]; then
    DISK_AVAIL=$(df -g . 2>/dev/null | awk 'NR==2 {print $4}')
fi

if [ -n "${DISK_AVAIL}" ] && [ "${DISK_AVAIL}" -gt 0 ] 2>/dev/null; then
    if [ "${DISK_AVAIL}" -lt 50 ]; then
        echo "  ⚠ ${DISK_AVAIL} GB available (recommend >100 GB for checkpoints)"
        WARNINGS=$((WARNINGS + 1))
    elif [ "${DISK_AVAIL}" -lt 100 ]; then
        echo "  ○ ${DISK_AVAIL} GB available (adequate, monitor during training)"
    else
        echo "  ✓ ${DISK_AVAIL} GB available"
    fi
else
    echo "  ○ Could not determine disk space"
fi
echo ""

# --- 4. HuggingFace Token ---
echo "─── HuggingFace Access ───"
if [ -n "${HF_TOKEN:-}" ] || [ -f ~/.cache/huggingface/token ]; then
    echo "  ✓ HF token found"
    # Quick check if we can access a gated model
    if python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.model_info('google/gemma-2-2b')
" 2>/dev/null; then
        echo "  ✓ Can access HF Hub (tested with public model)"
    else
        echo "  ⚠ HF Hub access may be limited"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ⚠ No HF token found (set HF_TOKEN or run huggingface-cli login)"
    echo "    Required for gated models (Gemma)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# --- 5. GCS Access (if bucket configured) ---
echo "─── GCS Access ───"
if [ -n "${GCS_BUCKET:-}" ]; then
    if command -v gsutil &>/dev/null; then
        if gsutil ls "${GCS_BUCKET}" &>/dev/null; then
            echo "  ✓ GCS bucket accessible: ${GCS_BUCKET}"
        else
            echo "  ⚠ Cannot access GCS bucket: ${GCS_BUCKET}"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "  ⚠ gsutil not installed (needed for checkpoint sync)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ○ GCS_BUCKET not set (checkpoint sync disabled)"
fi
echo ""

# --- 6. W&B Access ---
echo "─── Weights & Biases ───"
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "  ✓ WANDB_API_KEY set"
elif python3 -c "import wandb; assert wandb.api.api_key" 2>/dev/null; then
    echo "  ✓ W&B logged in (via wandb login)"
else
    echo "  ○ W&B not configured (logging will be local only)"
fi
echo ""

# --- 7. Config Validation ---
if [ -n "${CONFIG}" ]; then
    echo "─── Config Validation ───"
    if [ -f "${CONFIG}" ]; then
        echo "  ✓ Config file exists: ${CONFIG}"
        # Validate YAML can be parsed
        if python3 -c "from src.utils.config_utils import load_config; load_config('${CONFIG}')" 2>/dev/null; then
            echo "  ✓ Config YAML is valid"
        else
            echo "  ✗ Config YAML has errors"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo "  ✗ Config file not found: ${CONFIG}"
        ERRORS=$((ERRORS + 1))
    fi
    echo ""
fi

# --- Summary ---
echo "═══════════════════════════════════════════════════════════════"
if [ ${ERRORS} -gt 0 ]; then
    echo "  RESULT: FAIL (${ERRORS} errors, ${WARNINGS} warnings)"
    echo "  Fix errors before launching training."
    exit 1
elif [ ${WARNINGS} -gt 0 ]; then
    echo "  RESULT: PASS with warnings (${WARNINGS} warnings)"
    echo "  Training may work but check warnings above."
    exit 2
else
    echo "  RESULT: ALL CHECKS PASSED ✓"
    echo "  Ready to launch training."
    exit 0
fi
