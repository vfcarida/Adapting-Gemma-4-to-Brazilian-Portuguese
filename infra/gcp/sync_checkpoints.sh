#!/bin/bash
# =============================================================================
# Sync Checkpoints — GCS ↔ Local/VM
# =============================================================================
# Bidirectional sync of checkpoints, outputs, and reports between
# the training VM and Google Cloud Storage.
#
# Usage:
#   ./sync_checkpoints.sh upload           # VM → GCS (after training)
#   ./sync_checkpoints.sh download         # GCS → local (for analysis)
#   ./sync_checkpoints.sh watch            # Continuous sync every 30min
#   ./sync_checkpoints.sh status           # Show what's in GCS
# =============================================================================

set -euo pipefail

GCS_BUCKET="${GCS_BUCKET:-gs://gemma4-pt-br}"
LOCAL_DIR="${LOCAL_DIR:-./outputs}"
SYNC_INTERVAL="${SYNC_INTERVAL:-1800}"  # 30 minutes

case "${1:-status}" in
    upload)
        echo "Uploading outputs → ${GCS_BUCKET}/outputs/"
        gsutil -m rsync -r -x '.*\.pyc$|.*__pycache__.*' \
            outputs/ "${GCS_BUCKET}/outputs/"

        echo "Uploading reports → ${GCS_BUCKET}/reports/"
        gsutil -m rsync -r reports/ "${GCS_BUCKET}/reports/" 2>/dev/null || true

        echo ""
        echo "Upload complete. Contents:"
        gsutil du -sh "${GCS_BUCKET}/outputs/"
        ;;

    download)
        echo "Downloading ${GCS_BUCKET}/outputs/ → ${LOCAL_DIR}/"
        mkdir -p "${LOCAL_DIR}"
        gsutil -m rsync -r "${GCS_BUCKET}/outputs/" "${LOCAL_DIR}/"

        echo "Downloading reports..."
        mkdir -p reports
        gsutil -m rsync -r "${GCS_BUCKET}/reports/" reports/ 2>/dev/null || true

        echo ""
        echo "Download complete."
        du -sh "${LOCAL_DIR}"
        ;;

    download-results)
        echo "Downloading only eval results and reports (no checkpoints)..."
        mkdir -p reports "${LOCAL_DIR}/eval_cache"

        gsutil -m cp "${GCS_BUCKET}/outputs/eval_cache/*.json" "${LOCAL_DIR}/eval_cache/" 2>/dev/null || true
        gsutil -m rsync -r "${GCS_BUCKET}/reports/" reports/ 2>/dev/null || true

        echo "Results downloaded."
        ;;

    watch)
        echo "Watching for changes (sync every ${SYNC_INTERVAL}s)..."
        echo "Press Ctrl+C to stop."
        echo ""
        while true; do
            echo "[$(date -u +%H:%M:%S)] Syncing..."
            gsutil -m rsync -r -x '.*\.pyc$|.*__pycache__.*' \
                outputs/ "${GCS_BUCKET}/outputs/" 2>/dev/null || true
            gsutil -m rsync -r reports/ "${GCS_BUCKET}/reports/" 2>/dev/null || true
            echo "[$(date -u +%H:%M:%S)] Done. Next sync in ${SYNC_INTERVAL}s"
            sleep "${SYNC_INTERVAL}"
        done
        ;;

    status)
        echo "=== GCS Bucket Status: ${GCS_BUCKET} ==="
        echo ""
        echo "--- Outputs ---"
        gsutil ls "${GCS_BUCKET}/outputs/" 2>/dev/null || echo "  (empty)"
        echo ""
        echo "--- Checkpoints ---"
        gsutil ls "${GCS_BUCKET}/outputs/cpt_pilot/" 2>/dev/null || echo "  (no pilot checkpoints)"
        gsutil ls "${GCS_BUCKET}/outputs/cpt_main/" 2>/dev/null || echo "  (no main checkpoints)"
        echo ""
        echo "--- Reports ---"
        gsutil ls "${GCS_BUCKET}/reports/" 2>/dev/null || echo "  (no reports)"
        echo ""
        echo "--- Total Size ---"
        gsutil du -sh "${GCS_BUCKET}/" 2>/dev/null || echo "  Bucket not found"
        ;;

    clean-checkpoints)
        echo "WARNING: This will delete intermediate checkpoints (keeping only 'final' dirs)"
        echo "Press Ctrl+C to cancel, Enter to continue..."
        read -r

        # Remove checkpoint-XXXX dirs, keep only /final/
        for dir in cpt_pilot cpt_main; do
            echo "Cleaning ${GCS_BUCKET}/outputs/${dir}/checkpoint-*"
            gsutil -m rm -r "${GCS_BUCKET}/outputs/${dir}/checkpoint-*" 2>/dev/null || true
        done
        echo "Cleanup complete."
        ;;

    *)
        echo "Usage: $0 {upload|download|download-results|watch|status|clean-checkpoints}"
        exit 1
        ;;
esac
