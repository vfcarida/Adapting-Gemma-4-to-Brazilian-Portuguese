#!/bin/bash
# =============================================================================
# Submit Training Job — Gemma 4 Portuguese Adaptation
# =============================================================================
# Launches a training job on a GCP GPU VM. Supports both interactive (SSH)
# and background (tmux/nohup) modes.
#
# Usage:
#   ./submit_training_job.sh cpt_pilot              # CPT pilot (E4B, LoRA)
#   ./submit_training_job.sh cpt_main               # CPT main (26B, full)
#   ./submit_training_job.sh eval                   # Run evaluation suite
#   ./submit_training_job.sh merge                  # Residual merge sweep
#   ./submit_training_job.sh ablations              # Full ablation matrix
#
# The script:
#   1. SSHs into the training VM
#   2. Starts a tmux session for resilience
#   3. Runs the training command
#   4. Syncs outputs to GCS on completion
# =============================================================================

set -euo pipefail

# --- Configuration ---
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
GCS_BUCKET="${GCS_BUCKET:-gs://gemma4-pt-br}"
WORKSPACE="/workspace/repo"

# --- Parse arguments ---
JOB_TYPE="${1:-cpt_pilot}"
INSTANCE_NAME="${2:-gemma4-pt-br-pilot}"  # Override with second arg

case "${JOB_TYPE}" in
    cpt_pilot)
        TRAIN_CMD="gemma4pt train-cpt configs/train/cpt_pilot.yaml"
        SESSION_NAME="cpt-pilot"
        echo "Job: CPT Pilot (E4B, LoRA, 5B tokens)"
        ;;
    cpt_main)
        TRAIN_CMD="deepspeed --num_gpus=4 -m src.train.cpt_trainer --config configs/train/cpt_main.yaml"
        SESSION_NAME="cpt-main"
        INSTANCE_NAME="${2:-gemma4-pt-br-main}"
        echo "Job: CPT Main (26B, full fine-tune, DeepSpeed ZeRO-3)"
        ;;
    cpt_main_50b)
        TRAIN_CMD="deepspeed --num_gpus=4 -m src.train.cpt_trainer --config configs/train/cpt_main.yaml --override training.max_steps=-1"
        SESSION_NAME="cpt-main-50b"
        INSTANCE_NAME="${2:-gemma4-pt-br-main}"
        echo "Job: CPT Main 50B tokens (26B, full, extended)"
        ;;
    eval)
        TRAIN_CMD="gemma4pt eval --config configs/eval/benchmarks.yaml"
        SESSION_NAME="eval"
        echo "Job: Evaluation (all benchmarks, all models)"
        ;;
    merge)
        TRAIN_CMD="gemma4pt merge --base-model google/gemma-4-E4B --instruct-model google/gemma-4-E4B-it --cpt-model outputs/cpt_pilot/final --alpha 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 --method ties --density 0.6"
        SESSION_NAME="merge"
        echo "Job: Residual Merge (TIES, alpha sweep)"
        ;;
    ablations)
        TRAIN_CMD="bash scripts/run_ablations.sh"
        SESSION_NAME="ablations"
        echo "Job: Full ablation matrix"
        ;;
    sft)
        TRAIN_CMD="gemma4pt train-sft configs/train/sft.yaml"
        SESSION_NAME="sft"
        echo "Job: SFT"
        ;;
    *)
        echo "Usage: $0 {cpt_pilot|cpt_main|cpt_main_50b|eval|merge|ablations|sft} [instance-name]"
        exit 1
        ;;
esac

echo "Instance: ${INSTANCE_NAME}"
echo "Session:  ${SESSION_NAME}"
echo ""

# --- Build remote command ---
# The command runs inside tmux for resilience against SSH disconnection.
# On completion, it syncs outputs to GCS.
REMOTE_SCRIPT=$(cat <<'HEREDOC'
#!/bin/bash
set -euo pipefail

cd WORKSPACE_PLACEHOLDER

# Pull latest code
git pull origin main 2>/dev/null || true

# Activate environment
export CUDA_VISIBLE_DEVICES=CUDA_DEVICES_PLACEHOLDER
export WANDB_PROJECT="gemma4-pt-br"
export WANDB_RUN_GROUP="SESSION_PLACEHOLDER"

echo "=== Starting: SESSION_PLACEHOLDER ==="
echo "Command: COMMAND_PLACEHOLDER"
echo "Time: $(date -u)"
echo ""

# Run training
COMMAND_PLACEHOLDER

EXIT_CODE=$?

echo ""
echo "=== Completed: SESSION_PLACEHOLDER (exit=$EXIT_CODE) ==="
echo "Time: $(date -u)"

# Sync outputs to GCS
echo "Syncing outputs to GCS..."
gsutil -m rsync -r outputs/ BUCKET_PLACEHOLDER/outputs/
gsutil -m rsync -r reports/ BUCKET_PLACEHOLDER/reports/ 2>/dev/null || true

echo "Sync complete."
HEREDOC
)

# Replace placeholders
REMOTE_SCRIPT="${REMOTE_SCRIPT//WORKSPACE_PLACEHOLDER/${WORKSPACE}}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//COMMAND_PLACEHOLDER/${TRAIN_CMD}}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//SESSION_PLACEHOLDER/${SESSION_NAME}}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//BUCKET_PLACEHOLDER/${GCS_BUCKET}}"

# Set CUDA devices based on job type
if [[ "${JOB_TYPE}" == "cpt_main"* ]]; then
    REMOTE_SCRIPT="${REMOTE_SCRIPT//CUDA_DEVICES_PLACEHOLDER/0,1,2,3}"
else
    REMOTE_SCRIPT="${REMOTE_SCRIPT//CUDA_DEVICES_PLACEHOLDER/0}"
fi

# --- Submit job via SSH + tmux ---
echo "Submitting job to ${INSTANCE_NAME}..."
echo ""

gcloud compute ssh "${INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --command="
        # Kill existing session if any
        tmux kill-session -t ${SESSION_NAME} 2>/dev/null || true

        # Write script to file
        cat > /tmp/train_${SESSION_NAME}.sh << 'SCRIPT_EOF'
${REMOTE_SCRIPT}
SCRIPT_EOF
        chmod +x /tmp/train_${SESSION_NAME}.sh

        # Start in tmux
        tmux new-session -d -s ${SESSION_NAME} 'bash /tmp/train_${SESSION_NAME}.sh 2>&1 | tee /workspace/repo/outputs/${SESSION_NAME}.log'

        echo 'Job submitted in tmux session: ${SESSION_NAME}'
        echo ''
        echo 'To attach:  tmux attach -t ${SESSION_NAME}'
        echo 'To monitor: tail -f /workspace/repo/outputs/${SESSION_NAME}.log'
    "

echo ""
echo "Job submitted successfully!"
echo ""
echo "Monitor:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} -- 'tmux attach -t ${SESSION_NAME}'"
echo ""
echo "Check logs:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} -- 'tail -100 ${WORKSPACE}/outputs/${SESSION_NAME}.log'"
echo ""
echo "GPU utilization:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} -- 'nvidia-smi'"
