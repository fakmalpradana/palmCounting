#!/usr/bin/env bash
# setup-gpu-sa.sh — create and configure service account for GPU Cloud Run
# Usage: PROJECT_ID=palmcounting bash infra/setup-gpu-sa.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-palmcounting}"
SA_NAME="palm-counter-gpu"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Creating service account: ${SA_EMAIL}"

gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="Palm Counter GPU Worker" \
  --project="${PROJECT_ID}" || echo "Service account already exists"

# Allow GPU service to read/write GCS bucket
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"

# Allow GPU service to read/write Firestore
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/datastore.user"

echo "Service account configured."
echo "Next: run deploy-gpu.sh to deploy the GPU Cloud Run service."
