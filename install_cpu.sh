#!/usr/bin/env bash
# Install Palm Counter — CPU (macOS / Linux)
set -e
CONDA=$(conda info --base 2>/dev/null)/bin/conda
if [ ! -x "$CONDA" ]; then
  echo "❌ conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi
echo "▶ Creating conda environment 'palmcounting' (CPU)…"
"$CONDA" env create -f environment.yml --force
echo "✅ Done. Activate with: conda activate palmcounting"
echo "   Then start the server: uvicorn app.main:app --reload"
