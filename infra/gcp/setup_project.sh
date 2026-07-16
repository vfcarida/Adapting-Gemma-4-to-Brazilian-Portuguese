#!/bin/bash
# =============================================================================
# GCP Project Setup — One-time configuration
# =============================================================================
# Run this ONCE to set up the GCP project for Gemma 4 PT-BR training.
# Creates: GCS bucket, secrets, enables APIs, configures IAM.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Project created and billing enabled
#   - Your HuggingFace token and W&B API key ready
#
# Usage:
#   export GCP_PROJECT_ID="your-project-id"
#   export HF_TOKEN="hf_xxxxx"
#   export WANDB_API_KEY="xxxxx"
#   ./setup_project.sh
# =============================================================================

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-central1}"
BUCKET_NAME="${GCS_BUCKET_NAME:-gemma4-pt-br}"

echo "=== GCP Project Setup ==="
echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo "Bucket:  gs://${BUCKET_NAME}"
echo ""

# --- 1. Set project ---
echo "[1/6] Setting project..."
gcloud config set project "${PROJECT_ID}"

# --- 2. Enable required APIs ---
echo "[2/6] Enabling APIs..."
gcloud services enable \
    compute.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    aiplatform.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com

# --- 3. Create GCS bucket ---
echo "[3/6] Creating GCS bucket..."
gsutil mb -p "${PROJECT_ID}" -l "${REGION}" -c STANDARD "gs://${BUCKET_NAME}" 2>/dev/null || \
    echo "  Bucket already exists"

# Create directory structure
gsutil cp /dev/null "gs://${BUCKET_NAME}/data/.keep"
gsutil cp /dev/null "gs://${BUCKET_NAME}/outputs/.keep"
gsutil cp /dev/null "gs://${BUCKET_NAME}/reports/.keep"
gsutil cp /dev/null "gs://${BUCKET_NAME}/scripts/.keep"

# Upload startup script to bucket
gsutil cp infra/gcp/startup_script.sh "gs://${BUCKET_NAME}/scripts/startup_script.sh"

# --- 4. Create secrets ---
echo "[4/6] Creating secrets in Secret Manager..."

if [ -n "${HF_TOKEN:-}" ]; then
    echo -n "${HF_TOKEN}" | gcloud secrets create hf-token \
        --data-file=- --replication-policy=automatic 2>/dev/null || \
    echo -n "${HF_TOKEN}" | gcloud secrets versions add hf-token --data-file=-
    echo "  hf-token: created/updated"
else
    echo "  WARNING: HF_TOKEN not set, skipping"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    echo -n "${WANDB_API_KEY}" | gcloud secrets create wandb-api-key \
        --data-file=- --replication-policy=automatic 2>/dev/null || \
    echo -n "${WANDB_API_KEY}" | gcloud secrets versions add wandb-api-key --data-file=-
    echo "  wandb-api-key: created/updated"
else
    echo "  WARNING: WANDB_API_KEY not set, skipping"
fi

# --- 5. Configure IAM ---
echo "[5/6] Configuring IAM..."
# Get default compute service account
SA=$(gcloud iam service-accounts list --filter="displayName:Compute Engine" --format="value(email)" | head -1)

if [ -n "${SA}" ]; then
    # Grant access to secrets
    gcloud secrets add-iam-policy-binding hf-token \
        --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor" 2>/dev/null || true
    gcloud secrets add-iam-policy-binding wandb-api-key \
        --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor" 2>/dev/null || true
    echo "  Service account ${SA} granted access to secrets"
fi

# --- 6. Request GPU quota (informational) ---
echo "[6/6] GPU Quota check..."
echo ""
echo "IMPORTANT: You may need to request GPU quota increases:"
echo "  - NVIDIA_A100_80GB_GPUS: at least 4 (for main training)"
echo "  - PREEMPTIBLE_NVIDIA_A100_80GB_GPUS: at least 4 (for spot instances)"
echo ""
echo "Request at: https://console.cloud.google.com/iam-admin/quotas"
echo "Filter by: 'NVIDIA' in region '${REGION}'"
echo ""

# --- Done ---
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Request GPU quota (if needed)"
echo "  2. Upload Aurora-PT data: gsutil -m cp -r <local_data>/ gs://${BUCKET_NAME}/data/"
echo "  3. Create training VM: ./infra/gcp/create_instance.sh pilot"
echo "  4. Submit training job: ./infra/gcp/submit_training_job.sh cpt_pilot"
