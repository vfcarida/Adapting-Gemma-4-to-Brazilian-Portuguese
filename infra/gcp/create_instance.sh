#!/bin/bash
# =============================================================================
# Create GCP GPU Instance — Gemma 4 Portuguese Adaptation
# =============================================================================
# Creates a VM with GPU for training. Supports multiple configurations:
#   - pilot:   1x A100 80GB (CPT pilot, eval)
#   - main:    4x A100 80GB (CPT principal 26B)
#   - large:   8x H100 80GB (CPT 31B dense, large-scale)
#
# Usage:
#   ./create_instance.sh pilot        # 1x A100 for pilot experiments
#   ./create_instance.sh main         # 4x A100 for main CPT
#   ./create_instance.sh large        # 8x H100 for large experiments
#   ./create_instance.sh eval         # 1x A100 for evaluation only
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - GPU quota approved in your GCP project
#   - Secret Manager secrets created (hf-token, wandb-api-key)
# =============================================================================

set -euo pipefail

# --- Configuration ---
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
GCS_BUCKET="${GCS_BUCKET:-gs://gemma4-pt-br}"
INSTANCE_PREFIX="gemma4-pt-br"
REPO_BRANCH="${REPO_BRANCH:-main}"

# Deep Learning VM image
IMAGE_FAMILY="pytorch-2-4-cu124-debian-11"
IMAGE_PROJECT="deeplearning-platform-release"

# --- Parse argument ---
CONFIG="${1:-pilot}"

case "${CONFIG}" in
    pilot|eval)
        INSTANCE_NAME="${INSTANCE_PREFIX}-pilot"
        MACHINE_TYPE="a2-highgpu-1g"
        ACCELERATOR="type=nvidia-tesla-a100,count=1"
        BOOT_DISK_SIZE="200GB"
        LOCAL_SSD_COUNT=1
        SPOT="--provisioning-model=SPOT --instance-termination-action=STOP"
        echo "Config: PILOT — 1x A100 80GB (Spot)"
        ;;
    main)
        INSTANCE_NAME="${INSTANCE_PREFIX}-main"
        MACHINE_TYPE="a2-ultragpu-4g"
        ACCELERATOR="type=nvidia-a100-80gb,count=4"
        BOOT_DISK_SIZE="500GB"
        LOCAL_SSD_COUNT=2
        SPOT="--provisioning-model=SPOT --instance-termination-action=STOP"
        echo "Config: MAIN — 4x A100 80GB (Spot)"
        ;;
    main-ondemand)
        INSTANCE_NAME="${INSTANCE_PREFIX}-main-od"
        MACHINE_TYPE="a2-ultragpu-4g"
        ACCELERATOR="type=nvidia-a100-80gb,count=4"
        BOOT_DISK_SIZE="500GB"
        LOCAL_SSD_COUNT=2
        SPOT=""
        echo "Config: MAIN ON-DEMAND — 4x A100 80GB"
        ;;
    large)
        INSTANCE_NAME="${INSTANCE_PREFIX}-large"
        MACHINE_TYPE="a3-highgpu-8g"
        ACCELERATOR="type=nvidia-h100-80gb,count=8"
        BOOT_DISK_SIZE="1000GB"
        LOCAL_SSD_COUNT=4
        SPOT="--provisioning-model=SPOT --instance-termination-action=STOP"
        echo "Config: LARGE — 8x H100 80GB (Spot)"
        ;;
    *)
        echo "Usage: $0 {pilot|eval|main|main-ondemand|large}"
        exit 1
        ;;
esac

echo ""
echo "Project:  ${PROJECT_ID}"
echo "Zone:     ${ZONE}"
echo "Machine:  ${MACHINE_TYPE}"
echo "GPU:      ${ACCELERATOR}"
echo ""

# --- Create instance ---
echo "Creating instance: ${INSTANCE_NAME}..."

LOCAL_SSD_ARGS=""
for i in $(seq 1 ${LOCAL_SSD_COUNT}); do
    LOCAL_SSD_ARGS="${LOCAL_SSD_ARGS} --local-ssd=interface=NVME,size=375GB"
done

gcloud compute instances create "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="${ACCELERATOR}" \
    --maintenance-policy=TERMINATE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --boot-disk-size="${BOOT_DISK_SIZE}" \
    --boot-disk-type=pd-ssd \
    ${LOCAL_SSD_ARGS} \
    --scopes=cloud-platform \
    --metadata=startup-script-url="${GCS_BUCKET}/scripts/startup_script.sh",repo-branch="${REPO_BRANCH}" \
    --metadata-from-file=startup-script=infra/gcp/startup_script.sh \
    ${SPOT} \
    --labels=project=gemma4-pt-br,config="${CONFIG}"

echo ""
echo "Instance created: ${INSTANCE_NAME}"
echo ""
echo "Connect:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID}"
echo ""
echo "Monitor startup:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} -- 'tail -f /var/log/syslog | grep startup'"
echo ""
echo "Stop (to save costs):"
echo "  gcloud compute instances stop ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID}"
