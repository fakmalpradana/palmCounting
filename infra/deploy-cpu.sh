#!/usr/bin/env bash
# deploy-cpu.sh — update CPU Cloud Run service with new env vars
# Usage: PROJECT_ID=palmcounting GOOGLE_CLIENT_ID=... GOOGLE_CLIENT_SECRET=... JWT_SECRET=... CLEANUP_SECRET=... [GPU_WORKER_URL=<url from deploy-gpu.sh>] bash infra/deploy-cpu.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-palmcounting}"
REGION="asia-southeast1"
SERVICE="palm-counter"
: "${GOOGLE_CLIENT_ID:?GOOGLE_CLIENT_ID must be set}"
: "${GOOGLE_CLIENT_SECRET:?GOOGLE_CLIENT_SECRET must be set}"
: "${JWT_SECRET:?JWT_SECRET must be set}"
: "${CLEANUP_SECRET:?CLEANUP_SECRET must be set}"

echo "Updating CPU service: ${SERVICE}"

gcloud run services update "${SERVICE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
    --update-env-vars="^:^GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}:GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET}:JWT_SECRET=${JWT_SECRET}:CLEANUP_SECRET=${CLEANUP_SECRET}:JWT_ACCESS_EXPIRE_MINUTES=${JWT_ACCESS_EXPIRE_MINUTES:-15}:JWT_REFRESH_EXPIRE_DAYS=${JWT_REFRESH_EXPIRE_DAYS:-7}:FIRESTORE_PROJECT_ID=${PROJECT_ID}:FRONTEND_URL=${FRONTEND_URL:-https://palm-counter-981519809891.asia-southeast1.run.app}:GCS_BUCKET_NAME=${GCS_BUCKET_NAME:-palmcounting-uploads}:GPU_WORKER_URL=${GPU_WORKER_URL:-}:PYTHONUNBUFFERED=1"

echo "CPU service updated."
