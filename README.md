# 🏦 Institutional Volume Screener & ML Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An advanced Machine Learning platform designed to reverse-engineer institutional "Smart Money" footprints in the Indian Stock Market (NSE). This tool bypasses the "noise" of retail trading to find where big players are accumulating shares.

---

## 🎯 What this Platform Does
This project hunts for two specific types of institutional activity:
*   **The Action List:** Explosive breakouts with massive volume surges and price strength.
*   **The Stealth Watchlist:** Quiet, "spring-coiled" accumulation where the price is flat but delivery percentages are surging.

The system uses a **Walk-Forward ML Ensemble** (XGBoost, Random Forest, etc.) to predict the probability of a stock gaining >5% within the next 10 days.

---

## 🚀 Quick Start

### Windows Users
*   Ensure you have [Python 3.8+](https://www.python.org/downloads/) installed.
*   Download this repository.
*   Double-click **`run.bat`**.
*   The script will handle the environment setup, library installation, and data sync automatically.

### Linux/Mac Users
```bash
chmod +x run.sh
./run.sh
```

> **⚠️ Important Note on First Launch:**
> The first time you run this, the pipeline needs to download **5 years of historical data** from the NSE. Due to exchange rate limits, this process takes **1 to 2 hours**. Once downloaded, daily updates take only 2-3 minutes.

---

## 🏗️ Project Architecture
*   **`app.py`**: Interactive Streamlit dashboard for live alerts and backtesting.
*   **`pipeline.py`**: The engine that handles data fetching, feature engineering, and AI model training.
*   **`run.bat` / `run.sh`**: One-click launchers for different operating systems.
*   **`PLATFORM_GUIDE.md`**: Detailed documentation on the math and logic behind the screeners.

---

## 📊 Key Features
*   **Smart Money Score:** A proprietary 0-100 score based on Delivery Anomalies, Volatility Contraction (Squeezes), and Volume-Price Trends.
*   **Realistic Backtesting:** All profits are calculated from the **Next Day's Open price**, accounting for overnight gaps and execution reality.
*   **Market Regime Filter:** Analyze how the AI performs in Bull markets vs. Bear markets.
*   **Corporate Action Adjustment:** Automatically adjusts historical prices for stock splits and bonuses.

---

## 🛠️ Tech Stack
*   **Language:** Python 3.8+
*   **Machine Learning:** XGBoost, Scikit-Learn (Random Forest, SVM, MLP)
*   **Data Handling:** Pandas, PyArrow, Nselib
*   **UI/Dashboard:** Streamlit
*   **Financial APIs:** NSE (via Nselib) and Yahoo Finance

---

## ⚖️ Disclaimer
This software is for **educational and research purposes only**. It is not financial advice. Trading stocks involves significant risk of loss. Always consult with a certified financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred using this software.
