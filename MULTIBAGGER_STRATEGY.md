# 🏦 Institutional Edge: Momentum Multibagger Overhaul (The Holy Grail)

## 🎯 1. Mission Objective
To build a highly robust quantitative trading system that identifies "Smart Money" accumulation in the Indian Stock Market (NSE).
*   **Target:** >60% Out-of-Sample (Blind Test) Precision.
*   **Risk-Reward:** ~3:1 Ratio (+25% Target vs. ~9% Stop Loss).
*   **Frequency:** 4–5 high-conviction signals per month (Top 5% of all signals).
*   **Timeframe:** Momentum holding period of up to 60 trading days (~3 months).

---

## 🏗️ 2. Core Architecture
The system uses a **Walk-Forward ML Ensemble** trained on **10 years of historical data** (2015–2026). This ensures the model has "lived through" major market crises like the 2018 bear market, the 2020 COVID crash, and 2022's high-inflation volatility.

### A. Data Engine (`pipeline.py`)
*   **Input:** Daily NSE Bhavcopy with Delivery data (Crucial for tracking institutional holding).
*   **Corporate Actions:** Retroactive adjustment for all Splits and Bonuses to maintain smooth price/volume curves.
*   **Purity Filter:** Hard exclusion of all ETFs, Funds, Commodities, and Debt instruments to ensure 100% Equity data.

### B. Machine Learning Brain
*   **Ensemble:** Voting Classifier using the top 2 performers among **XGBoost, LightGBM, RandomForest, and CatBoost**.
*   **Validation:** Strict Walk-Forward Cross-Validation by year.
*   **Blind Test Buffer:** The model is strictly forbidden from training on the **last 6 months (126 days)** of data. This period is reserved for true "Blind Validation" on the dashboard.

---

## 🧪 3. Technical Feature Engineering (The Edge)
To achieve >60% accuracy, the system uses advanced features to filter "Noise" from "Smart Money":

1.  **Smart Money Score (0–100):** A composite of delivery anomalies, volume trends, and Bollinger Band squeeze metrics.
2.  **Retail Trap Filter (`Upper_Wick_Ratio`):** Calculates the ratio of the upper wick to the candle range. High volume surges with long upper wicks (rejections) are penalized to avoid fake breakouts.
3.  **Trend Score:** A weighted composite of Relative Strength (RS) over 20/126 days and EMA alignment (50/200).
4.  **Market Regime Filter:** Incorporates **India VIX** and **Market Breadth** (% of stocks above 50-EMA) to ensure the model only trades when the macro "tide" is rising.
5.  **Leakage Fix:** Market Breadth is shifted by 1 day (`.shift(1)`) so the model predicts "tomorrow" using only "yesterday's" breadth.

---

## 📏 4. Trade Management Logic
The AI is trained on a specific "Momentum Multibagger" profile:

*   **Entry:** Next-Day Open (T+1) after a signal.
*   **Stop Loss (The Shield):** **3.0x ATR** (Average True Range). This wide stop is designed to survive "shakeouts" and volatility. Capped between 3% and 15%.
*   **Target (The Goal):** **+25% Gain**.
*   **Holding Period:** 60 trading days.
*   **Labeling:** A signal is marked "Success" (1) if it hits +25% *before* hitting the stop loss within 60 days.

---

## 📊 5. Dashboard & Execution
*   **Top 5% Filter:** The Live Screener uses dynamic thresholds (95th percentile of historical scores) to only show the "cream of the crop" setups.
*   **Blind Period Badge:** All signals from the last 6 months are clearly marked "BLIND TEST" to allow immediate verification of real-world performance.
*   **Position Sizing:** Built-in risk manager to calculate share quantity based on a 1% risk-per-trade rule.

---

## 🛠️ 6. Operational Commands
*   **Full Sync & Train:** `./run.sh`
*   **Fix Missing Data (Backfill):** `env BACKFILL_MISSING_DATES=1 ./run.sh`
*   **Dashboard Only:** `streamlit run app.py`

**The strategy is built on the mathematical principle that high-conviction institutional entries, when given room to breathe (3x ATR), result in asymmetric multibagger returns significantly more often than they fail.**
