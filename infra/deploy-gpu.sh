#!/usr/bin/env bash
# deploy-gpu.sh — deploy GPU Cloud Run service for commercial tier
# Usage: PROJECT_ID=palmcounting bash infra/deploy-gpu.sh
#
# Prerequisites:
#   - GPU quota approved for asia-southeast1 (nvidia-l4)
#   - GPU Cloud Run service image built and pushed
#   - Service account with Storage + Firestore access (see setup-gpu-sa.sh)

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-palmcounting}"
REGION="asia-southeast1"
SERVICE="palm-counter-gpu"
IMAGE="asia-southeast1-docker.pkg.dev/${PROJECT_ID}/palm-counting-repo/palm-counter-gpu:latest"
SA="palm-counter-gpu@${PROJECT_ID}.iam.gserviceaccount.com"

: "${CLEANUP_SECRET:?CLEANUP_SECRET must be set}"

echo "Deploying GPU service: ${SERVICE}"

gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --no-allow-unauthenticated \
  --service-account="${SA}" \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --memory=32Gi \
  --cpu=8 \
  --min-instances=0 \
  --max-instances=3 \
  --timeout=3600 \
    --set-env-vars="^:^FIRESTORE_PROJECT_ID=${PROJECT_ID}:GCS_BUCKET_NAME=${GCS_BUCKET_NAME:-palmcounting-uploads}:CLEANUP_SECRET=${CLEANUP_SECRET}:PYTHONUNBUFFERED=1"

echo "GPU service deployed."
GPU_URL=$(gcloud run services describe "${SERVICE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")
echo "GPU service URL: ${GPU_URL}"
echo "Add this as GPU_WORKER_URL in your CPU service env vars."
