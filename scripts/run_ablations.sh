#!/bin/bash
# =============================================================================
# Run Ablations — Full experimental matrix for Gemma 4 PT-BR adaptation
# =============================================================================
# Executes the pilot ablation experiments sequentially (or in parallel if
# multiple GPUs are available). Each experiment saves to its own output dir
# and logs are captured independently.
#
# Usage:
#   bash scripts/run_ablations.sh                    # Run all pilot ablations
#   bash scripts/run_ablations.sh --group B          # Only PEFT method comparison
#   bash scripts/run_ablations.sh --group C          # Only replay ratio sweep
#   bash scripts/run_ablations.sh --group D          # Only residual merge
#   bash scripts/run_ablations.sh --group E          # Only SFT
#   bash scripts/run_ablations.sh --parallel 4       # Run 4 experiments in parallel
#   bash scripts/run_ablations.sh --dry-run          # Show what would run
#
# Prerequisites:
#   - pip install -e ".[all]"
#   - GPU available (at least 1x A100 80GB for pilot experiments)
#   - Aurora-PT data accessible (HF Hub or local)
#   - gemma4pt preflight passes
# =============================================================================

set -euo pipefail

# --- Configuration ---
OUTPUT_BASE="outputs/ablations"
CONFIG_BASE="configs/train/cpt_pilot.yaml"
EVAL_CONFIG="configs/eval/benchmarks.yaml"
LOG_DIR="${OUTPUT_BASE}/logs"
PARALLEL=1
GROUP="all"
DRY_RUN=false
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --group) GROUP="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --output) OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}"

echo "=== Gemma 4 PT-BR Ablation Runner ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Group:     ${GROUP}"
echo "Parallel:  ${PARALLEL}"
echo "Output:    ${OUTPUT_BASE}"
echo ""

# --- Helper functions ---
run_experiment() {
    local NAME="$1"
    local OVERRIDES="$2"
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    local LOG_FILE="${LOG_DIR}/${NAME}_${TIMESTAMP}.log"

    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY-RUN] ${NAME}"
        echo "  Config: ${CONFIG_BASE}"
        echo "  Overrides: ${OVERRIDES}"
        echo "  Output: ${OUTPUT_DIR}"
        echo ""
        return 0
    fi

    echo "[$(date +%H:%M:%S)] Starting: ${NAME}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "  Log: ${LOG_FILE}"

    mkdir -p "${OUTPUT_DIR}"

    gemma4pt train-cpt "${CONFIG_BASE}" \
        --override output.output_dir="${OUTPUT_DIR}" \
        ${OVERRIDES} \
        2>&1 | tee "${LOG_FILE}"

    local EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] COMPLETED: ${NAME}"
    else
        echo "[$(date +%H:%M:%S)] FAILED: ${NAME} (exit=${EXIT_CODE})"
    fi

    return ${EXIT_CODE}
}

run_eval() {
    local NAME="$1"
    local MODEL_PATH="$2"
    local LOG_FILE="${LOG_DIR}/eval_${NAME}_${TIMESTAMP}.log"

    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY-RUN] Eval: ${NAME} (model=${MODEL_PATH})"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] Evaluating: ${NAME}"

    gemma4pt eval --config "${EVAL_CONFIG}" --model "${MODEL_PATH}" \
        2>&1 | tee "${LOG_FILE}"
}

run_merge() {
    local NAME="$1"
    local CPT_PATH="$2"
    local METHOD="$3"
    local DENSITY="$4"
    local ALPHAS="$5"

    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY-RUN] Merge: ${NAME} (method=${METHOD}, density=${DENSITY})"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] Merging: ${NAME} (method=${METHOD})"

    gemma4pt merge \
        --base-model google/gemma-4-E4B \
        --instruct-model google/gemma-4-E4B-it \
        --cpt-model "${CPT_PATH}" \
        --alpha ${ALPHAS} \
        --method "${METHOD}" \
        --density "${DENSITY}" \
        --output-dir "${OUTPUT_BASE}/${NAME}"
}

# =============================================================================
# GRUPO B: PEFT Method Comparison (LoRA vs DoRA vs QLoRA)
# =============================================================================
run_group_B() {
    echo "=== Group B: PEFT Method Comparison ==="
    echo ""

    # B1: LoRA r=64
    run_experiment "B1_cpt_lora" \
        "--override training.peft_method=lora lora.r=64 data_mixture=pt_en_15"

    # B2: DoRA r=64
    run_experiment "B2_cpt_dora" \
        "--override training.peft_method=dora lora.r=64 lora.use_dora=true data_mixture=pt_en_15"

    # B3: QLoRA r=64
    run_experiment "B3_cpt_qlora" \
        "--override training.peft_method=qlora lora.r=64 data_mixture=pt_en_15"
}

# =============================================================================
# GRUPO C: Replay Ratio Sweep (0%, 5%, 10%, 15%)
# =============================================================================
run_group_C() {
    echo "=== Group C: Replay Ratio Sweep ==="
    echo ""

    # C1: 100% Portuguese (no replay)
    run_experiment "C1_cpt_pt_only" \
        "--override data_mixture=pt_only"

    # C2: 95% PT + 5% EN
    run_experiment "C2_cpt_pt_en_5" \
        "--override data_mixture=pt_en_5"

    # C3: 90% PT + 10% EN
    run_experiment "C3_cpt_pt_en_10" \
        "--override data_mixture=pt_en_10"

    # C4: 85% PT + 15% EN (default)
    run_experiment "C4_cpt_pt_en_15" \
        "--override data_mixture=pt_en_15"
}

# =============================================================================
# GRUPO D: Residual Merge (multiple methods and alpha sweep)
# =============================================================================
run_group_D() {
    echo "=== Group D: Residual Merge ==="
    echo ""

    # Use best PEFT result from Group B (default: B1 LoRA)
    local CPT_PATH="${OUTPUT_BASE}/B1_cpt_lora/final"

    if [ ! -d "${CPT_PATH}" ]; then
        echo "ERROR: CPT model not found at ${CPT_PATH}"
        echo "Run Group B first: bash scripts/run_ablations.sh --group B"
        return 1
    fi

    # D1: Task Arithmetic (alpha sweep)
    run_merge "D1_task_arithmetic" "${CPT_PATH}" "task_arithmetic" "1.0" \
        "0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2"

    # D2: TIES-Merge (density 0.6, alpha sweep)
    run_merge "D2_ties" "${CPT_PATH}" "ties" "0.6" \
        "0.6 0.8 1.0 1.2"

    # D3: DARE-TIES (density 0.5, alpha sweep)
    run_merge "D3_dare_ties" "${CPT_PATH}" "dare_ties" "0.5" \
        "0.6 0.8 1.0 1.2"

    # D4: DARE-Linear (density 0.5, alpha sweep)
    run_merge "D4_dare_linear" "${CPT_PATH}" "dare_linear" "0.5" \
        "0.6 0.8 1.0 1.2"
}

# =============================================================================
# GRUPO E: SFT after CPT
# =============================================================================
run_group_E() {
    echo "=== Group E: SFT ==="
    echo ""

    local CPT_PATH="${OUTPUT_BASE}/B1_cpt_lora/final"

    if [ ! -d "${CPT_PATH}" ]; then
        echo "ERROR: CPT model not found at ${CPT_PATH}"
        echo "Run Group B first."
        return 1
    fi

    # E1: SFT with PT instruction data
    gemma4pt train-sft configs/train/sft.yaml \
        --override model.base_id="${CPT_PATH}" output.output_dir="${OUTPUT_BASE}/E1_sft_pt" \
        2>&1 | tee "${LOG_DIR}/E1_sft_pt_${TIMESTAMP}.log"
}

# =============================================================================
# EVALUATION: Run eval on all completed experiments
# =============================================================================
run_evaluation() {
    echo "=== Running Evaluation Suite ==="
    echo ""

    # Find all 'final' directories
    for final_dir in ${OUTPUT_BASE}/*/final; do
        if [ -d "${final_dir}" ]; then
            local exp_name=$(basename $(dirname "${final_dir}"))
            run_eval "${exp_name}" "${final_dir}"
        fi
    done

    # Evaluate merge results (each alpha)
    for merge_dir in ${OUTPUT_BASE}/D*/alpha_*; do
        if [ -d "${merge_dir}" ]; then
            local exp_name=$(echo "${merge_dir}" | sed "s|${OUTPUT_BASE}/||" | tr '/' '_')
            run_eval "${exp_name}" "${merge_dir}"
        fi
    done
}

# =============================================================================
# Main execution
# =============================================================================
case "${GROUP}" in
    all)
        run_group_B
        run_group_C
        run_group_D
        run_group_E
        run_evaluation
        ;;
    B|b) run_group_B ;;
    C|c) run_group_C ;;
    D|d) run_group_D ;;
    E|e) run_group_E ;;
    eval) run_evaluation ;;
    *)
        echo "Unknown group: ${GROUP}"
        echo "Options: all, B, C, D, E, eval"
        exit 1
        ;;
esac

echo ""
echo "=== Ablation run complete ==="
echo "Results: ${OUTPUT_BASE}"
echo "Logs:    ${LOG_DIR}"
