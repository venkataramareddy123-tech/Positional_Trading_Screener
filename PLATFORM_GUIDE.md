# 🏦 Institutional Volume Screener & ML Research Platform
**Complete Architecture and User Guide**

Welcome to the Institutional Volume Screener! This platform is designed to reverse-engineer institutional footprints in the stock market. It operates by fetching raw market data, mathematically identifying hidden accumulation, training Machine Learning algorithms on historical patterns, and serving the results in a live portfolio tracker.

---

## 📁 1. Project Setup & File Structure
To run this program successfully, you only need the core source files. Here is what each file does:

*   **`app.py`**: The "Face" of the project. It runs the interactive Streamlit dashboard.
*   **`pipeline.py`**: The "Engine". It downloads data, calculates indicators (Volume surges, RS, Smart Money Score), and trains the Machine Learning models.
*   **`run.sh`**: The "Starter" for Linux/Mac. A bash script that sets up your environment and starts the program.
*   **`run.bat`**: The "Starter" for Windows. Double-click this to install and run everything automatically.
*   **`requirements.txt`**: The "Ingredients". Lists all the Python libraries (like `xgboost`, `streamlit`, `nselib`) needed for the app.
*   **`.gitignore`**: The "Shield". Tells Git to ignore large data files and local environments so your GitHub stays clean.
*   **`PLATFORM_GUIDE.md`**: The "Manual". This document!

### Why is `.gitignore` important?
The `.gitignore` file ensures that **heavy files** (like the `data/` folder which stores years of stock history), **log files** (temporary diagnostic notes), and **local files** (like `.venv/` and `__pycache__`) are not pushed to GitHub. This keeps your repository fast, lightweight, and professional.

---

## 🏗️ 2. The Core Architecture
The platform is built on a 3-stage architecture:
1. **The Data Engine (`pipeline.py`):** Automatically downloads daily EOD (End of Day) data from the NSE, adjusts for corporate actions (splits/bonuses), and engineers advanced technical metrics for every stock.
2. **The Machine Learning Brain (`pipeline.py`):** A walk-forward cross-validation engine that orchestrates a shootout between 5 different ML models to find the top 2 best-performing algorithms for the current market regime. 
3. **The Live Dashboard (`app.py`):** A reactive Streamlit UI that serves as both a live screener for tomorrow's trades and a historical backtesting playground.

---

## 🎯 3. The Twin Screeners
The platform hunts for two completely different trading setups, displayed side-by-side on the Live Screener tab.

### A. The Action List (Explosive Breakouts)
*The Action List hunts for stocks that are breaking out **today**. These are momentum plays ready for immediate execution.*
To trigger an Action List alert, a stock must strictly meet these 3 rules:
1. **Volume Surge > 2.0x:** The stock traded more than double its 30-day average volume OR double its 30-day average delivery volume.
2. **Close > VWAP:** The stock must close the day above its Volume Weighted Average Price, proving that buyers maintained control until the closing bell.
3. **Top 10% of Daily Range:** The closing price must be in the absolute top 10% of the daily candle (High minus Low).

### B. The Stealth Watchlist (Smart Money)
*The Stealth Watchlist hunts for stocks that show massive institutional accumulation behind the scenes without a price breakout yet.*
It triggers when a stock scores **80+ (out of 100)** on the proprietary Smart Money Algorithm:
1. **Delivery Anomaly (30 pts):** Surge in the 5-day Delivery % vs 50-day baseline.
2. **Volume-Price Trend (30 pts):** Higher volume on Green days than Red days.
3. **Volatility Contraction (30 pts):** Narrowing Bollinger Bands (The Squeeze).
4. **50-EMA Defense (10 pts):** Staying above the institutional support line.

---

## 🤖 4. The Machine Learning Framework
*   **The Goal:** Predict if a stock will gain **>5% within 10 days**.
*   **Realistic Entry:** Profit is calculated from the **Next Day's Open price** (T+1), accounting for overnight gaps.
*   **The Models:** Tests XGBoost, Random Forest, SVM, Neural Networks, and Logistic Regression.
*   **The Ensemble:** Automatically selects the Top 2 algorithms and averages their scores for the final **"ML Score (%)"**.

---

## 🚀 5. Getting Started (Installation & Usage)

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your machine.

### 2. First-Time Launch
**On Windows:**
Simply double-click **`run.bat`**.

**On Linux/Mac:**
Run the following in your terminal:
```bash
chmod +x run.sh
./run.sh
```

These scripts will:
*   Create a virtual environment (`.venv`).
*   Install all necessary libraries.
*   Download historical data (Note: The first run takes **1-2 hours** to fetch 5 years of NSE data).
*   Train the AI models.
*   Open the Dashboard in your browser.

### 3. Regular Use
Every evening after the market closes (around 7:00 PM IST), you can run `./run.sh` to sync the latest daily data and get fresh predictions for the next trading day.

---

## 📊 6. Troubleshooting
*   **Pipeline Failed?** Ensure you have an active internet connection. The NSE server can sometimes block requests if they are too frequent; the script handles this, but extreme cases might require a restart.
*   **Missing Data?** If your dashboard shows "Data not found," click the **Sync & Update** button in the **Data Management** tab of the app.
