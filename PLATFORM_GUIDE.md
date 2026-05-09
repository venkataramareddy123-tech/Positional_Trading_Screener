# 🏦 Institutional Volume Screener & ML Research Platform
**Complete Architecture and User Guide**

Welcome to the Institutional Volume Screener! This platform is designed to reverse-engineer institutional footprints in the stock market. It operates by fetching raw market data, mathematically identifying hidden accumulation, training Machine Learning algorithms on historical patterns, and serving the results in an interactive dashboard.

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
2. **The Machine Learning Brain (`pipeline.py`):** A walk-forward cross-validation engine that orchestrates a shootout between different ML models to find the top performing algorithms for the current market regime. 
3. **The Live Dashboard (`app.py`):** A reactive Streamlit UI that serves as both a live screener for tomorrow's trades and a historical backtesting playground.

---

## 🎯 3. The Twin Screeners
The platform hunts for two completely different trading setups, displayed side-by-side on the Live Screener tab.

### A. The Action List (Explosive Breakouts)
*The Action List hunts for stocks that are breaking out **today**. These are momentum plays ready for immediate execution.*
To trigger an Action List alert, a stock must meet core institutional criteria:
1. **Volume Surge > 1.5x:** The stock traded more than 50% above its 30-day average volume OR its 30-day average delivery volume.
2. **Close > VWAP:** The stock must close the day above its Volume Weighted Average Price, proving that buyers maintained control until the closing bell.
3. **Top 10% of Daily Range:** The closing price must be in the absolute top 10% of the daily candle (High minus Low).

### B. The Stealth Watchlist (Smart Money)
*The Stealth Watchlist hunts for stocks that show massive institutional accumulation behind the scenes without a price breakout yet.*
It triggers when a stock scores **high** on the proprietary Smart Money Algorithm:
1. **Delivery Anomaly:** Surge in the 5-day Delivery % vs 50-day baseline.
2. **Volume-Price Trend:** Higher volume on Green days than Red days.
3. **Volatility Contraction:** Narrowing Bollinger Bands (The Squeeze).
4. **50-EMA Defense:** Staying above the institutional support line.

---

## 🤖 4. The Machine Learning Framework
The AI doesn't just "guess"; it evaluates patterns against 5 years of historical outcomes using a rigorous **Walk-Forward validation** process.

*   **The Goal:** Predict if a stock will hit a **+15% target** within 20 trading days.
*   **Realistic Entry:** All metrics are calculated from the **Next Day's Open price** (T+1), ensuring the AI accounts for overnight gaps.
*   **The Models:** The system runs a "tournament" between **XGBoost, RandomForest, LightGBM, and CatBoost**.
*   **The Ensemble:** The platform automatically selects the **top 2 algorithms** based on their Out-of-Sample "Average Precision" and creates a weighted ensemble.
*   **Risk-Adjusted Logic:** The system uses a **2.5:1 Reward-to-Risk ratio** (targeting 15% gains vs. a ~6% ATR-based stop loss).

---

## 🛡️ 5. Risk Management System
Unlike simple screeners, this platform includes a professional-grade Risk Manager:
1.  **ATR-Based Stops:** Stop losses are dynamically set at **2.0x the Average True Range (ATR)**, ensuring the stock has enough "room to breathe" based on its own volatility.
2.  **Hard Survival Rules:** Stop losses are capped between **3% and 10%** to prevent catastrophic losses on volatile stocks.
3.  **Position Sizing:** The dashboard automatically calculates exactly how many shares to buy based on your **Total Capital** and a **1% Risk-per-Trade** rule.

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
