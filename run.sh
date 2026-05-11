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

# You can change DAYS_TO_FETCH for different historical lookbacks.
# 2520 = 10 Years (Recommended for Momentum Multibagger Strategy)
# 1250 = 5 Years
export DAYS_TO_FETCH=2520
export BACKFILL_MISSING_DATES=${BACKFILL_MISSING_DATES:-0}
export CUDA_VISIBLE_DEVICES=""

echo "Syncing NSE data, corporate actions, features, and models..."
./.venv/bin/python pipeline.py

echo "Starting Streamlit Dashboard..."
./.venv/bin/streamlit run app.py
