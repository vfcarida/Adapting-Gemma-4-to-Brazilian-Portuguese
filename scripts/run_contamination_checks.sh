#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Run Contamination Checks                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "═══ Three-Tier Contamination Check ═══"

python -m src.data.contamination_checks \
    --dataset_id "${AURORA_DATASET_ID:-Itau-Unibanco/Aurora-PT}" \
    --threshold 0.8 \
    --num_samples "${CONTAM_NUM_SAMPLES:-10000}" \
    --output "reports/contamination_report.json" \
    --ref_datasets \
        "eduagarcia/enem_challenge:test:alternatives" \
        "eduagarcia-temp/BLUEX:test:alternatives" \
        "nilc-nlp/assin2:test:hypothesis" \
        "eduagarcia/oab_exams:test:alternatives" \
        "ruanchaves/hatebr:test:instagram_comment" \
        "ruanchaves/tweetsentbr:test:tweet_text"

echo "✓ Contamination check complete — see reports/contamination_report.json"
