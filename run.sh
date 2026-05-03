#!/bin/bash
set -e
# Setup script for Institutional Volume Screener

if [ ! -d ".venv" ]; then
    echo "Setting up Python virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing required packages..."
pip install -r requirements.txt

# You can change DAYS_TO_FETCH to 1250 for a full 5-year run.
# The default is 1250, but it will take a while to download.
export DAYS_TO_FETCH=1250
export BACKFILL_MISSING_DATES=${BACKFILL_MISSING_DATES:-0}
export CUDA_VISIBLE_DEVICES=""

echo "Syncing NSE data, corporate actions, features, and models..."
python pipeline.py

echo "Starting Streamlit Dashboard..."
.venv/bin/streamlit run app.py
