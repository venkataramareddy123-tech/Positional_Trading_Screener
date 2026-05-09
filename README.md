# 🏦 Institutional Volume Screener & ML Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An advanced Machine Learning platform designed to reverse-engineer institutional "Smart Money" footprints in the Indian Stock Market (NSE). This tool bypasses the "noise" of retail trading to find where big players are accumulating shares using Volume-Price Analysis and Walk-Forward ML validation.

---

## 🎯 What this Platform Does
This project hunts for institutional activity using two core strategies:
*   **The Action List:** Explosive breakouts with massive volume surges (>1.5x) and price strength.
*   **The Stealth Watchlist:** Quiet, "spring-coiled" accumulation where delivery percentages are surging while price remains stable.

The system uses a **Walk-Forward ML Ensemble** (XGBoost, Random Forest, LightGBM, CatBoost) to predict the probability of a stock hitting a **+15% target** within 20 days.

---

## 🏗️ Project Architecture
*   **`app.py`**: Interactive Streamlit dashboard for live signals, risk management, and ML diagnostics.
*   **`pipeline.py`**: The high-performance engine for data fetching (NSE), feature engineering (30+ indicators), and walk-forward model training.
*   **`PLATFORM_GUIDE.md`**: Detailed technical documentation on the math and logic behind the screeners.
*   **`run.bat` / `run.sh`**: One-click launchers for automated environment setup and execution.

---

## 📊 Key Features
*   **Smart Money Score:** A proprietary 0-100 score based on Delivery Anomalies, Volatility Contraction (Squeezes), and Volume-Price Trends.
*   **Walk-Forward Validation:** Unlike standard backtests, this evaluates the AI on "unseen" future years to ensure the strategy is robust against changing market regimes.
*   **Institutional Risk Management:** Integrated position sizing based on account capital, risk-per-trade (1%), and ATR-based stop losses.
*   **Next-Day Execution:** All performance metrics are calculated using the **Next Day's Open price**, ensuring results are achievable in real-world trading.

---

## 🚀 Quick Start

### Windows
1. Install [Python 3.8+](https://www.python.org/downloads/).
2. Double-click **`run.bat`**. The script handles environment setup, dependencies, and data sync automatically.

### Linux / Mac
```bash
chmod +x run.sh
./run.sh
```

> **⚠️ Note on First Launch:** The pipeline downloads **5 years of historical data** from the NSE. Due to exchange rate limits, this takes **~1 hour**. Subsequent daily updates take only 2-3 minutes.

---

## 🛠️ Tech Stack
*   **ML:** XGBoost, Random Forest, LightGBM, CatBoost, Scikit-Learn
*   **Data:** Pandas, PyArrow, Nselib, YFinance
*   **UI:** Streamlit (Custom Dark Theme)

---

## ⚖️ Disclaimer
This software is for **educational and research purposes only**. Trading stocks involves significant risk. The "ML Score" is a statistical probability, not a guarantee. Always consult a certified financial advisor. The authors are not responsible for any financial losses.
