#!/usr/bin/env bash
# Activate the palmcounting conda env and start the server
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate palmcounting
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
