#!/bin/bash
# =============================================================================
# GCP VM Startup Script — Gemma 4 Portuguese Adaptation
# =============================================================================
# This script runs automatically when the VM starts.
# It sets up the environment, clones the repo, and prepares for training.
#
# Usage: Attached as metadata startup-script when creating VM instances.
# =============================================================================

set -euo pipefail

# --- Configuration (override via instance metadata) ---
PROJECT_BUCKET="${GCS_BUCKET:-gs://gemma4-pt-br}"
REPO_URL="${REPO_URL:-https://github.com/vfcarida/Adapting-Gemma-4-to-Brazilian-Portuguese.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORKSPACE="/workspace"
DATA_DIR="/mnt/data"
CONDA_DIR="/opt/conda"

echo "=== Gemma 4 PT-BR: VM Startup ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Instance: $(hostname)"

# --- 1. Install NVIDIA drivers (if not already present) ---
if ! command -v nvidia-smi &> /dev/null; then
    echo "[1/7] Installing NVIDIA drivers..."
    /opt/deeplearning/install-driver.sh || true
else
    echo "[1/7] NVIDIA drivers already installed"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# --- 2. Install system dependencies ---
echo "[2/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git tmux htop tree jq

# --- 3. Setup workspace ---
echo "[3/7] Setting up workspace..."
mkdir -p "${WORKSPACE}" "${DATA_DIR}"

# Mount local SSD if available (for fast I/O during training)
if [ -b /dev/nvme0n1 ] && ! mount | grep -q "${DATA_DIR}"; then
    echo "  Formatting and mounting local SSD..."
    mkfs.ext4 -F /dev/nvme0n1
    mount /dev/nvme0n1 "${DATA_DIR}"
    chmod 777 "${DATA_DIR}"
fi

# --- 4. Clone repository ---
echo "[4/7] Cloning repository..."
if [ -d "${WORKSPACE}/repo" ]; then
    cd "${WORKSPACE}/repo"
    git fetch origin
    git checkout "${REPO_BRANCH}"
    git pull origin "${REPO_BRANCH}"
else
    git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${WORKSPACE}/repo"
    cd "${WORKSPACE}/repo"
fi

# --- 5. Setup Python environment ---
echo "[5/7] Setting up Python environment..."
if [ -d "${CONDA_DIR}" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate base
fi

pip install --quiet --upgrade pip setuptools wheel
pip install --quiet -e ".[all]"

# --- 6. Authenticate with HuggingFace and W&B ---
echo "[6/7] Configuring credentials..."

# HF Token from Secret Manager (or instance metadata)
HF_TOKEN=$(gcloud secrets versions access latest --secret="hf-token" 2>/dev/null || \
    curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token" \
    -H "Metadata-Flavor: Google" 2>/dev/null || echo "")

if [ -n "${HF_TOKEN}" ]; then
    echo "  HuggingFace token configured"
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
    export HF_TOKEN
fi

# W&B API Key
WANDB_KEY=$(gcloud secrets versions access latest --secret="wandb-api-key" 2>/dev/null || echo "")
if [ -n "${WANDB_KEY}" ]; then
    echo "  W&B configured"
    export WANDB_API_KEY="${WANDB_KEY}"
    wandb login "${WANDB_KEY}" 2>/dev/null || true
fi

# --- 7. Sync data from GCS ---
echo "[7/7] Syncing data from GCS..."
gsutil -m rsync -r "${PROJECT_BUCKET}/data/" "${DATA_DIR}/" 2>/dev/null || \
    echo "  No data in GCS yet (first run)"

# Create output dirs
mkdir -p "${WORKSPACE}/repo/outputs"
ln -sf "${DATA_DIR}" "${WORKSPACE}/repo/data" 2>/dev/null || true

# --- Done ---
echo ""
echo "=== Setup Complete ==="
echo "Workspace: ${WORKSPACE}/repo"
echo "Data:      ${DATA_DIR}"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  No GPU"
echo ""
echo "Next steps:"
echo "  cd ${WORKSPACE}/repo"
echo "  gemma4pt preflight"
echo "  gemma4pt train-cpt configs/train/cpt_pilot.yaml"
