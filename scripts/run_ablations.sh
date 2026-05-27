#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Run Automated Ablations                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "═══ Running Ablations ═══"
mkdir -p "$PROJECT_ROOT/ablations"

# A) Gemma 4 base sem CPT (Baseline is already covered in baselines script)
# B) Gemma 4 base + CPT (Pure Aurora)
echo "▶ Ablation B: Pure CPT"
python -m src.train.cpt_trainer --config configs/cpt_pilot.yml --override "data.en_ratio=0.0" "training.output_dir=ablations/B_cpt_pure"

# C) Gemma 4 base + CPT + replay em inglês
echo "▶ Ablation C: CPT + EN Replay"
python -m src.train.cpt_trainer --config configs/cpt_pilot.yml --override "data.en_ratio=0.15" "training.output_dir=ablations/C_cpt_replay"

# D) Gemma 4 base + CPT + residual merge
echo "▶ Ablation D: CPT + Residual Merge"
python -m src.train.residual_merge --config configs/merge.yml --alpha 1.0 --override "merge.cpt_model_path=ablations/C_cpt_replay/final" "merge.output_dir=ablations/D_cpt_merge"

# E) Gemma 4 base + CPT + SFT pt-BR
echo "▶ Ablation E: CPT + SFT PT-BR"
python -m src.train.sft_trainer --config configs/sft.yml --override "model.model_id=ablations/C_cpt_replay/final" "training.output_dir=ablations/E_cpt_sft_pt"

# F) Gemma 4 base + CPT + SFT pt-BR+EN
echo "▶ Ablation F: CPT + SFT PT/EN Mix"
# Requires a mixed instruction dataset configured
python -m src.train.sft_trainer --config configs/sft.yml --override "model.model_id=ablations/C_cpt_replay/final" "data.dataset_id=SET_MIXED_DATASET_ID" "training.output_dir=ablations/F_cpt_sft_mix"

echo "✓ Ablations completed. See ablations/ directory."
