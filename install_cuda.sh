#!/usr/bin/env bash
# Install Palm Counter — CUDA (Linux / WSL2 on Windows)
# Requires: CUDA Toolkit 11.8+ and cuDNN installed system-wide
set -e
CONDA=$(conda info --base 2>/dev/null)/bin/conda
if [ ! -x "$CONDA" ]; then
  echo "❌ conda not found. Install Miniconda first."
  exit 1
fi
echo "▶ Creating conda environment 'palmcounting-cuda'…"
"$CONDA" env create -f environment-cuda.yml --force
echo "✅ Done. Activate with: conda activate palmcounting-cuda"
echo "   Then start the server: uvicorn app.main:app --reload"
