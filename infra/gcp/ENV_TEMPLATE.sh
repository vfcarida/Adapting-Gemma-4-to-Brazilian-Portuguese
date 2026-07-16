#!/bin/bash
# =============================================================================
# Environment variables for GCP training
# =============================================================================
# Copy this file and fill in your values:
#   cp ENV_TEMPLATE.sh .env.gcp
#   source .env.gcp
#
# NEVER commit .env.gcp to git!
# =============================================================================

# GCP Project
export GCP_PROJECT_ID="your-gcp-project-id"
export GCP_REGION="us-central1"
export GCP_ZONE="us-central1-a"
export GCS_BUCKET="gs://gemma4-pt-br"
export GCS_BUCKET_NAME="gemma4-pt-br"

# Repository
export REPO_URL="https://github.com/vfcarida/Adapting-Gemma-4-to-Brazilian-Portuguese.git"
export REPO_BRANCH="main"

# Credentials (obtain from respective services)
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Training
export WANDB_PROJECT="gemma4-pt-br"
export WANDB_ENTITY=""  # Your W&B team/user
