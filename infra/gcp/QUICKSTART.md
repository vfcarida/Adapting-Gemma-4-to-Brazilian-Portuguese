# GCP Training Quickstart

## Prerequisites

1. GCP account with billing enabled
2. `gcloud` CLI installed and authenticated
3. GPU quota approved (request at Console → IAM → Quotas)
4. HuggingFace token with access to `google/gemma-4` models
5. (Optional) Weights & Biases account

## First-time Setup

```bash
# 1. Configure environment
cp infra/gcp/ENV_TEMPLATE.sh .env.gcp
# Edit .env.gcp with your values
source .env.gcp

# 2. Setup GCP project (creates bucket, secrets, enables APIs)
./infra/gcp/setup_project.sh

# 3. Verify setup
gcloud compute machine-types list --filter="zone:${GCP_ZONE}" | grep -E "a2|a3"
```

## Running Experiments

### Pilot (1x A100 80GB, ~$120 for 24h)

```bash
# Create VM
./infra/gcp/create_instance.sh pilot

# SSH into VM
gcloud compute ssh gemma4-pt-br-pilot --zone=${GCP_ZONE}

# Inside VM: run preflight + pilot training
cd /workspace/repo
gemma4pt preflight
gemma4pt train-cpt configs/train/cpt_pilot.yaml

# Or submit as background job (from local):
./infra/gcp/submit_training_job.sh cpt_pilot
```

### Main Training (4x A100 80GB, ~$4000-6500 for 3-5 days)

```bash
# Create main VM
./infra/gcp/create_instance.sh main

# Submit CPT main job
./infra/gcp/submit_training_job.sh cpt_main gemma4-pt-br-main
```

### Evaluation (1x A100, ~$60 for 12h)

```bash
./infra/gcp/submit_training_job.sh eval gemma4-pt-br-pilot
```

## Monitoring

```bash
# Check instance status and costs
./infra/gcp/stop_and_cleanup.sh list
./infra/gcp/stop_and_cleanup.sh cost

# SSH and attach to training session
gcloud compute ssh gemma4-pt-br-pilot --zone=${GCP_ZONE} -- 'tmux attach -t cpt-pilot'

# Download results locally
./infra/gcp/sync_checkpoints.sh download-results
```

## Cost Management

```bash
# Stop instances when not training
./infra/gcp/stop_and_cleanup.sh stop

# Sync checkpoints before stopping
./infra/gcp/sync_checkpoints.sh upload

# Clean intermediate checkpoints (keep only final)
./infra/gcp/sync_checkpoints.sh clean-checkpoints
```

## Estimated Costs (Spot pricing)

| Phase | Config | Duration | Cost |
|-------|--------|----------|------|
| Pilot (1 variant) | 1x A100 | 24h | ~$50 |
| Pilot (7 variants) | 7x A100 parallel | 24h | ~$350 |
| Main 20B tokens | 4x A100 | 3-5 days | ~$2,000 |
| Main 50B tokens | 4x A100 | 7-12 days | ~$5,000 |
| Evaluation | 1x A100 | 12-24h | ~$50-100 |
| Merge sweep | 1x A100 | 2h | ~$10 |

**Total recommended budget: $3,000-8,000 (Spot instances)**
