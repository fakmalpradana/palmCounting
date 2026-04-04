# ============================================================
# Palm Counter — Optimized Dockerfile for GCP Cloud Run
# Copyright © 2026 Geo AI Twinverse.
# Contributors: Fikri Kurniawan, Fairuz Akmal Pradana
# ============================================================
#
# Strategy:
#   • python:3.11-slim (Debian bookworm) — minimal footprint
#   • System GDAL/GEOS/PROJ from apt-get — no compilation needed
#   • pip deps installed in a separate layer for build caching
#   • Non-root USER for GCP security best practices
#   • PORT env var respected (Cloud Run injects 8080 at runtime)
#
# Model weights (~28 MB) are pre-fetched from GCS by Cloud Build
# (see cloudbuild.yaml) and placed in /workspace/models/ before
# docker build runs. COPY . . picks them up from there.
# ============================================================

FROM python:3.11-slim

# Suppress apt interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level geospatial and OpenCV dependencies
# Package names verified for Debian bookworm (python:3.11-slim base)
RUN apt-get update && apt-get install -y --no-install-recommends \
        # GDAL / GEOS / PROJ runtime libs
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        # rtree spatial index
        libspatialindex-dev \
        # OpenCV headless runtime
        libgl1 \
        libglib2.0-0 \
        # Build tools (needed for some pip packages)
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (separate layer → cache-friendly)
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir -r requirements.txt

# Explicitly bake in the default model weights so the container is
# self-contained and survives Cloud Run scale-to-zero without a GCS fetch.
# This must be copied BEFORE the general COPY so it is never overwritten by
# an empty directory from the build context (Cloud Build clears /models/ root
# but never touches app/models/default/).
COPY app/models/ ./app/models/

# Copy full application source
# Note: models/ (root-level) is pre-populated by Cloud Build step "fetch-model"
# which downloads from gs://palmcounting-models/models/ before docker build.
COPY . .

# Ensure writable runtime directories exist and owned by nobody
RUN mkdir -p uploads results output \
 && chown -R nobody:nogroup uploads results output /app

# Run as non-root for GCP security policy
USER nobody

# Cloud Run injects PORT env var at runtime (default 8080)
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Startup — shell form so $PORT expands at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120 --log-level info
