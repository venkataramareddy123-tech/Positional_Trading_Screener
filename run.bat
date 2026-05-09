@echo off
SETLOCAL EnableDelayedExpansion

:: Setup script for Institutional Volume Screener (Windows)

echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python from python.org and try again.
    pause
    exit /b
)

if not exist .venv (
    echo Setting up Python virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing required packages (this may take a minute)...
pip install -r requirements.txt --quiet

:: Set Environment Variables
set DAYS_TO_FETCH=1250
set BACKFILL_MISSING_DATES=0
set CUDA_VISIBLE_DEVICES=""

echo.
echo ==========================================================
echo Syncing NSE data, corporate actions, and training AI...
echo (Note: The first run can take 1-2 hours due to NSE limits)
echo ==========================================================
python pipeline.py

echo.
echo Starting Streamlit Dashboard...
python -m streamlit run app.py

pause
