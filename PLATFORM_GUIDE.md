# 🏦 Institutional Volume Screener & ML Research Platform
**Complete Architecture and User Guide**

Welcome to the Institutional Volume Screener! This platform is designed to reverse-engineer institutional footprints in the stock market. It operates by fetching raw market data, mathematically identifying hidden accumulation, training Machine Learning algorithms on historical patterns, and serving the results in a live portfolio tracker.

This document serves as a complete breakdown of how the platform works and what every filter and metric means.

---

## 🏗️ 1. The Core Architecture
The platform is built on a 3-stage architecture:
1. **The Data Engine (`pipeline.py`):** Automatically downloads daily EOD (End of Day) data from the NSE, adjusts for corporate actions (splits/bonuses), and engineers advanced technical metrics for every stock.
2. **The Machine Learning Brain (`pipeline.py`):** A walk-forward cross-validation engine that orchestrates a shootout between 5 different ML models to find the top 2 best-performing algorithms for the current market regime. 
3. **The Live Dashboard (`app.py`):** A reactive Streamlit UI that serves as both a live screener for tomorrow's trades and a historical backtesting playground.

---

## 🎯 2. The Twin Screeners
The platform hunts for two completely different trading setups, displayed side-by-side on the Live Screener tab.

### A. The Action List (Explosive Breakouts)
*The Action List hunts for stocks that are breaking out **today**. These are momentum plays ready for immediate execution.*
To trigger an Action List alert, a stock must strictly meet these 3 rules:
1. **Volume Surge > 2.0x:** The stock traded more than double its 30-day average volume OR double its 30-day average delivery volume.
2. **Close > VWAP:** The stock must close the day above its Volume Weighted Average Price, proving that buyers maintained control until the closing bell.
3. **Top 10% of Daily Range:** The closing price must be in the absolute top 10% of the daily candle (High minus Low). This filters out stocks that spiked but faced heavy selling pressure in the afternoon.

### B. The Stealth Watchlist (Smart Money)
*The Stealth Watchlist hunts for stocks that are completely flat and boring, but show massive institutional accumulation behind the scenes. These are "spring-coiled" setups.*
To trigger a Stealth alert, the stock must NOT be breaking out today, and it must score an **80 or higher (out of 100)** on the proprietary Smart Money Algorithm. The 100 points are calculated as follows:
1. **Delivery Anomaly (30 pts):** A sudden surge in the 5-day average Delivery Percentage compared to the 50-day baseline. Institutions are physically locking up the float.
2. **Volume-Price Trend (30 pts):** Over the last 15 days, green (up) days must command significantly higher volume than red (down) days.
3. **Volatility Contraction (30 pts):** The Bollinger Band Width is ranked against the last 6 months. Highest points are awarded when the bands are at their tightest, flat-lining range.
4. **50-EMA Defense (10 pts):** The stock must strictly hold above its 50-day Exponential Moving Average.

---

## 🤖 3. The Machine Learning Framework
When you run the ML Pipeline, the system doesn't just guess; it rigorously trains itself on your historical data.

*   **The Goal (Target):** The AI tries to predict if a stock will generate a **Max Return of >5% within 10 days**.
*   **Realistic Entry Pricing:** To ensure backtesting isn't fake, the system mathematically forces the entry price to be the **Open of the next trading day** (T+1 Open). If a stock gaps up 3% overnight, the AI calculates your profit from the gap-up price, not yesterday's close.
*   **Walk-Forward Validation:** The AI splits your 5 years of data into chronological chunks. It trains on 2021 to predict 2022, trains on 2022 to predict 2023, etc. This prevents "lookahead bias."
*   **The Ensemble:** It tests XGBoost, Random Forest, SVM, Neural Networks, and Logistic Regression. It automatically selects the Top 2 models and averages their predictions to generate the final **"ML Score (%)"**.

---

## 📊 4. The Historical Analytics Dashboard
The "Historical Analytics" tab allows you to backtest the AI's performance over custom timeframes (e.g., Last 1 Month, Last 1 Year). 

**Key Metrics Explained:**
*   **Target Holding Period Slider (1-30 Days):** Dynamically adjust how long you plan to hold the stock. All metrics on the page will instantly recalculate for that exact holding duration.
*   **Market Regime Filter:** 
    *   *Bull Market:* Only analyzes trades taken when Nifty > 50-day EMA.
    *   *Bear Market:* Only analyzes trades taken when Nifty < 50-day EMA. (You will often notice a higher profit factor in Bear Markets because only the highest-conviction institutional accumulation survives heavy market selloffs).
*   **Profit Factor:** The ultimate risk-reward metric. Calculated as `Average Profit / Average Drawdown`. A Profit Factor of 2.0 means the stock historically makes twice as much profit as it experiences in temporary drawdowns.
*   **Symbol Hit Rate Leaderboard:** Automatically ranks the most historically reliable stocks by their Profit Factor. Stocks require a minimum of 2 historical triggers to qualify for the board.

---

## 🚀 5. Getting Started (First-Time Run)
If you are running `run.sh` for the very first time on a new machine:
1. The script will automatically build an isolated Python environment and install all dependencies.
2. The pipeline will begin downloading 5 years (1250 days) of NSE data. **This will take 1-2 hours depending on NSE rate limits.**
3. During data engineering, you may see `PerformanceWarning: DataFrame is highly fragmented`. This is normal and harmless.
4. Once the data is processed and the AI is trained, the Streamlit Dashboard will automatically launch in your browser.
