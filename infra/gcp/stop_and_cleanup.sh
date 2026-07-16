#!/bin/bash
# =============================================================================
# Stop & Cleanup — Cost management for GCP instances
# =============================================================================
# Stops or deletes training VMs to avoid unnecessary costs.
#
# Usage:
#   ./stop_and_cleanup.sh stop           # Stop all gemma4 instances
#   ./stop_and_cleanup.sh stop pilot     # Stop only pilot instance
#   ./stop_and_cleanup.sh delete pilot   # Delete pilot instance
#   ./stop_and_cleanup.sh cost           # Show current running costs
#   ./stop_and_cleanup.sh list           # List all gemma4 instances
# =============================================================================

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_PREFIX="gemma4-pt-br"

case "${1:-list}" in
    list)
        echo "=== Gemma 4 PT-BR Instances ==="
        gcloud compute instances list \
            --project="${PROJECT_ID}" \
            --filter="name~${INSTANCE_PREFIX}" \
            --format="table(name,zone,machineType.basename(),status,scheduling.provisioningModel)"
        ;;

    stop)
        TARGET="${2:-all}"
        if [ "${TARGET}" = "all" ]; then
            echo "Stopping all ${INSTANCE_PREFIX}-* instances..."
            INSTANCES=$(gcloud compute instances list \
                --project="${PROJECT_ID}" \
                --filter="name~${INSTANCE_PREFIX} AND status=RUNNING" \
                --format="value(name,zone)")

            if [ -z "${INSTANCES}" ]; then
                echo "No running instances found."
                exit 0
            fi

            echo "${INSTANCES}" | while read -r name zone; do
                echo "  Stopping ${name} (${zone})..."
                gcloud compute instances stop "${name}" --zone="${zone}" --project="${PROJECT_ID}" --async
            done
        else
            INSTANCE="${INSTANCE_PREFIX}-${TARGET}"
            echo "Stopping ${INSTANCE}..."
            gcloud compute instances stop "${INSTANCE}" --zone="${ZONE}" --project="${PROJECT_ID}"
        fi
        echo "Done."
        ;;

    delete)
        TARGET="${2:?Specify instance suffix: pilot, main, large}"
        INSTANCE="${INSTANCE_PREFIX}-${TARGET}"
        echo "WARNING: Deleting ${INSTANCE} (this is irreversible)"
        echo "Make sure outputs are synced to GCS first!"
        echo "Press Ctrl+C to cancel, Enter to continue..."
        read -r

        gcloud compute instances delete "${INSTANCE}" \
            --zone="${ZONE}" \
            --project="${PROJECT_ID}" \
            --quiet
        echo "Deleted: ${INSTANCE}"
        ;;

    cost)
        echo "=== Running Costs (approximate) ==="
        echo ""
        INSTANCES=$(gcloud compute instances list \
            --project="${PROJECT_ID}" \
            --filter="name~${INSTANCE_PREFIX} AND status=RUNNING" \
            --format="csv[no-heading](name,machineType.basename())")

        if [ -z "${INSTANCES}" ]; then
            echo "No running instances. Current cost: \$0/hr"
            exit 0
        fi

        TOTAL=0
        echo "${INSTANCES}" | while IFS=',' read -r name machine; do
            case "${machine}" in
                *a2-highgpu-1g*) COST="3.67" ;;
                *a2-ultragpu-4g*) COST="40.22" ;;
                *a3-highgpu-8g*) COST="80.44" ;;
                *) COST="?" ;;
            esac
            echo "  ${name} (${machine}): ~\$${COST}/hr"
        done

        echo ""
        echo "Tip: Use 'stop' when not actively training to save costs"
        ;;

    *)
        echo "Usage: $0 {list|stop [target]|delete <target>|cost}"
        exit 1
        ;;
esac
