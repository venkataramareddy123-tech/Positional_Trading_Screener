#!/bin/bash
# Setup script for Institutional Volume Screener

echo "Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing required packages..."
pip install -r requirements.txt

# You can change DAYS_TO_FETCH to 1250 for a full 5-year run.
# The default is 1250, but it will take a while to download.
export DAYS_TO_FETCH=1250
export CUDA_VISIBLE_DEVICES=""

echo "Running Data Pipeline (fetching $DAYS_TO_FETCH days of NSE data, this will take some time)..."
python pipeline.py

echo "Starting Streamlit Dashboard..."
streamlit run app.py
