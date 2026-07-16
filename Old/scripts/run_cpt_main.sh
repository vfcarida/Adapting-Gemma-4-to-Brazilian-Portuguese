#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Run CPT Main — Gemma-4-26B-A4B (Multi-GPU + DeepSpeed ZeRO-3)     ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# Auto-detect GPU count if not set
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"

echo "═══ Continued Pretraining (Main) — Gemma-4-26B-A4B ═══"
echo "GPUs: $NUM_GPUS"
echo "DeepSpeed: ZeRO-3 with CPU offloading"

# Multi-GPU with DeepSpeed
accelerate launch \
    --num_processes "$NUM_GPUS" \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_zero3.json \
    -m src.train.cpt_trainer \
    --config configs/cpt_main.yml \
    "$@"

echo "✓ CPT main complete — checkpoint at output/cpt_main_26b/final/"
