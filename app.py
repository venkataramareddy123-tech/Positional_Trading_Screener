import os

# Keep this dashboard on CPU so XGBoost does not keep the NVIDIA GPU awake.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import subprocess
import time
import json
from datetime import datetime

st.set_page_config(layout="wide", page_title="Institutional Edge Trader", page_icon="🏦")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #58a6ff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: transparent;
        border-radius: 4px 4px 0px 0px; padding: 10px 20px;
        color: #8b949e;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(88, 166, 255, 0.1);
        border-bottom: 2px solid #58a6ff !important; color: #58a6ff;
    }
    
    /* Signal Card Styling */
    .signal-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .signal-card:hover {
        transform: translateY(-4px);
        border-color: #58a6ff;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 10px;
    }
    .symbol-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    .signal-date {
        font-size: 0.85rem;
        color: #8b949e;
        background: #21262d;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-top: 15px;
    }
    .metric-box {
        background: rgba(255,255,255,0.03);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c9d1d9;
    }
    .score-badge {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .score-high { color: #3fb950; background: rgba(63, 185, 80, 0.1); border: 1px solid rgba(63, 185, 80, 0.2); }
    .score-med { color: #d29922; background: rgba(210, 153, 34, 0.1); border: 1px solid rgba(210, 153, 34, 0.2); }
    .score-low { color: #f85149; background: rgba(248, 81, 73, 0.1); border: 1px solid rgba(248, 81, 73, 0.2); }
    
    .level-container {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #30363d;
    }
    .level-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 0.9rem;
    }
    .target-val { color: #3fb950; font-weight: 600; }
    .stop-val { color: #f85149; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_data.parquet")
MODEL_FILE = os.path.join(DATA_DIR, "xgb_model.pkl")
METRICS_FILE = os.path.join(DATA_DIR, "ml_metrics.json")
LABEL_DIAGNOSTICS_FILE = os.path.join(DATA_DIR, "label_diagnostics.json")

# Features for prediction - Order must match pipeline.py exactly
MODEL_FEATURES = [
    'Vol_Surge', 'Del_Surge', 'Close_Location', 'Upper_Wick_Ratio',
    'RS_1D', 'RS_5D', 'RS_20D', 'RS_126D', 'ATR_30', 'BB_Width',
    'Nifty_vs_EMA50', 'Nifty_vs_EMA200', 'Dist_From_52W_High',
    'Smart_Money_Score', 'Is_Inst_Buy', 'Inst_Intensity', 
    'Del_Anomaly', 'Vol_Trend', 'Dist_From_EMA50',
    'India_VIX', 'VIX_Change_5D', 'VIX_Rank_252D',
    'Market_Breadth_50EMA', 'Trend_Score',
    'RSI_4', 'Dist_From_20SMA', 'BB_Squeeze_30', 'Price_Momentum_3M'
]

@st.cache_data
def load_data(mtime):
    if not os.path.exists(PROCESSED_FILE) or not os.path.exists(MODEL_FILE):
        return None, None, None
    
    # Load only necessary columns for the dashboard to save memory and time
    cols_to_load = [
        'Date', 'Symbol', 'Close', 'Open', 'High', 'Low', 'Volume', 
        'Delivery_Volume', 'First_Hit', 'Breakout_Signal',
        'Entry_Price', 'EMA_50'
    ]
    # Add model features and remove any duplicates
    for f in MODEL_FEATURES:
        if f not in cols_to_load:
            cols_to_load.append(f)
    
    # Use a faster engine and only load what's needed
    df = pd.read_parquet(PROCESSED_FILE, columns=cols_to_load)
    
    # Pre-calculate a mask for signals to make dashboard filtering instant
    latest_date = df['Date'].max()
    cutoff_date = latest_date - pd.Timedelta(days=730) # Keep 2 years of history for UI depth
    
    df = df[(df['First_Hit'] == True) | (df['Date'] >= cutoff_date)].copy()
    
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    metrics = None
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
    return df, model, metrics

def calculate_levels(row):
    """
    Implements the Multibagger Dynamic Execution Protocol.
    - Captures massive institutional moves by holding as long as the trend remains.
    - Stop Loss: Closing below the 50-day EMA or a hard -8% stop.
    """
    prev_close = row['Close']
    ema50 = row['EMA_50']
    
    # Gap-Up Limit: Max 3% gap up allowed for entry
    max_acceptable_open = prev_close * 1.03
    
    if pd.notna(row['Entry_Price']):
        entry = row['Entry_Price']
    else:
        entry = prev_close # Theoretical

    actual_entry = min(entry, max_acceptable_open)
    
    # Trailing Stop is the current EMA 50
    # Hard stop is -8% from entry
    hard_stop = actual_entry * 0.92
    trailing_stop = max(ema50, hard_stop)
        
    return pd.Series([actual_entry, trailing_stop])

mtime = os.path.getmtime(PROCESSED_FILE) if os.path.exists(PROCESSED_FILE) else 0
df, model, metrics = load_data(mtime)

# --- HEADER ---
st.title("💰 Multibagger Alpha Machine")
if df is not None:
    latest_date = df['Date'].max()
    # Explicitly calculate the 6-month blind cutoff for UI display
    blind_cutoff_date = latest_date - pd.Timedelta(days=126)
    st.caption(f"Strategy: Stage 2 Superperformance | Data: 10Y NSE Cash | Target: 2x-10x Gains")
else:
    st.warning("No data found. Please run the pipeline.")
    if st.button("🚀 Run Pipeline Now"):
        # run pipeline logic
        pass
    st.stop()

# --- TABS ---
tab1, tab2 = st.tabs(["🎯 Live Screener", "⚙️ System Configuration"])

with tab1:
    st.markdown("### 🏹 Active Trading Signals")
    
    # --- RISK MANAGER SECTION ---
    with st.expander("🛡️ Risk & Position Sizing Manager", expanded=True):
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            total_cap = st.number_input("Total Trading Capital (₹)", min_value=10000, value=100000, step=10000)
        with col_r2:
            risk_pct = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        with col_r3:
            risk_amt = total_cap * (risk_pct / 100)
            st.metric("Risk Amount per Trade", f"₹{risk_amt:,.0f}")
    
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        quality_filter = st.selectbox("Signal Quality Filter", options=["All Signals", "Alpha Engine (Top 5%)", "High Conviction (Top 10%)", "Solid Setup (Top 25%)", "Standard (Above Average)"], index=1)
    with col_f2:
        session_opts = {
            "Last Session": 1, 
            "Last 3 Sessions": 3, 
            "Last 5 Sessions": 5, 
            "Last 10 Sessions": 10, 
            "Last 20 Sessions": 20,
            "Last 60 Sessions (~3 Months)": 60,
            "Last 126 Sessions (~6 Months Blind)": 126,
            "Last 252 Sessions (~1 Year)": 252,
            "Last 504 Sessions (~2 Years)": 504,
            "All Time": 999999
        }
        selected_session_label = st.selectbox("Trading Sessions Lookback", list(session_opts.keys()), index=2)
        lookback_sessions = session_opts[selected_session_label]
    
    # Get the actual trading dates for the lookback
    available_dates = sorted(df['Date'].unique())
    start_date = available_dates[-lookback_sessions] if len(available_dates) >= lookback_sessions else available_dates[0]
    
    # Filter signals
    signals = df[(df['Date'] >= start_date) & (df['First_Hit'] == True)].copy()
    
    if not signals.empty:
        # Calculate levels
        signals[['Entry', 'Trailing_Stop']] = signals.apply(calculate_levels, axis=1)
        
        # Calculate Position Sizing
        # For Multibagger, we use a fixed 20% allocation as per the audit
        signals['Investment'] = total_cap * 0.20
        signals['Qty'] = (signals['Investment'] / (signals['Entry'] + 1e-8)).astype(int)
        
        # Predict ML Score
        X = signals[MODEL_FEATURES].fillna(0)
        signals['ML_Score'] = model.predict_proba(X)[:, 1] * 100
        
        # Calculate Dynamic Thresholds based on a Rolling Window (Last 252 Sessions)
        # This is the "All-Weather" engine fix: We compare leaders to the current market distribution.
        hist_signals = df[df['First_Hit'] == True].copy()
        hist_X = hist_signals[MODEL_FEATURES].fillna(0)
        hist_signals['ML_Score_Hist'] = model.predict_proba(hist_X)[:, 1] * 100
        
        # Get the threshold from the last 1 year of screener hits
        latest_hist_date = hist_signals['Date'].max()
        window_start = latest_hist_date - pd.Timedelta(days=365)
        recent_scores = hist_signals[hist_signals['Date'] >= window_start]['ML_Score_Hist']
        
        if not recent_scores.empty:
            top_5_thresh = np.percentile(recent_scores, 95)
            top_10_thresh = np.percentile(recent_scores, 90)
            top_25_thresh = np.percentile(recent_scores, 75)
            avg_thresh = np.mean(recent_scores)
        else:
            # Fallback to global if window is empty
            top_5_thresh = np.percentile(hist_signals['ML_Score_Hist'], 95)
            top_10_thresh = np.percentile(hist_signals['ML_Score_Hist'], 90)
            top_25_thresh = np.percentile(hist_signals['ML_Score_Hist'], 75)
            avg_thresh = np.mean(hist_signals['ML_Score_Hist'])
        
        # Apply Quality Filter using Dynamic Tiers
        if quality_filter == "Alpha Engine (Top 5%)":
            signals = signals[signals['ML_Score'] >= top_5_thresh]
        elif quality_filter == "High Conviction (Top 10%)":
            signals = signals[signals['ML_Score'] >= top_10_thresh]
        elif quality_filter == "Solid Setup (Top 25%)":
            signals = signals[signals['ML_Score'] >= top_25_thresh]
        elif quality_filter == "Standard (Above Average)":
            signals = signals[signals['ML_Score'] >= avg_thresh]
            
        signals = signals.sort_values(['Date', 'ML_Score'], ascending=[False, False])
        
        # --- RENDER SIGNAL CARDS ---
        cols_per_row = 3
        for i in range(0, len(signals), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(signals):
                    sig = signals.iloc[i + j]
                    
                    # Determine Score Class
                    score = sig['ML_Score']
                    score_class = "score-high" if score > 65 else ("score-med" if score > 45 else "score-low")
                    
                    # Calculate actual stop percentage for display
                    stop_pct = (1 - sig['Trailing_Stop'] / sig['Entry']) * 100
                    
                    with row_cols[j]:
                        # Check if signal is from the blind test period
                        is_blind = sig['Date'] > blind_cutoff_date
                        blind_badge = '<div style="background: #f85149; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.65rem; margin-top: 5px; display: inline-block;">BLIND TEST</div>' if is_blind else ""
                        
                        # Determine Engine Type
                        engine_type = "🚀 MULTIBAGGER"
                        engine_color = "#f0883e"
                        
                        st.markdown(f"""
                        <div class="signal-card">
                            <div class="card-header">
                                <div style="display: flex; flex-direction: column;">
                                    <span class="symbol-name">{sig['Symbol']}</span>
                                    <div style="background: {engine_color}; color: black; padding: 2px 6px; border-radius: 4px; font-size: 0.6rem; margin-top: 4px; font-weight: 800; display: inline-block;">{engine_type}</div>
                                    {blind_badge}
                                </div>
                                <span class="signal-date">{sig['Date'].strftime('%d %b')}</span>
                            </div>
                            <div style="text-align: center; margin-bottom: 20px;">
                                <div class="metric-label">AI Multibagger Score</div>
                                <div class="score-badge {score_class}">{score:.1f}%</div>
                            </div>
                            <div class="metric-grid">
                                <div class="metric-box">
                                    <div class="metric-label">Entry Price</div>
                                    <div class="metric-value">₹{sig['Entry']:,.2f}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">Portfolio Load</div>
                                    <div class="metric-value">20%</div>
                                </div>
                            </div>
                            <div class="level-container">
                                <div class="level-row">
                                    <span>Trailing Stop (EMA50)</span>
                                    <span class="stop-val">₹{sig['Trailing_Stop']:,.2f}</span>
                                </div>
                                <div class="level-row">
                                    <span>Current Risk</span>
                                    <span class="stop-val">-{stop_pct:.1f}%</span>
                                </div>
                                <div class="level-row" style="margin-top: 10px; border-top: 1px dashed #30363d; padding-top: 10px;">
                                    <span style="color: #8b949e;">Holding Period</span>
                                    <span style="color: #ffffff; font-weight: 700;">Trend-Following</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with st.expander("🔍 View Raw Signal Data Grid", expanded=False):
            display_df = signals[['Date', 'Symbol', 'ML_Score', 'Entry', 'Qty', 'Investment', 'Trailing_Stop', 'Smart_Money_Score']].copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%d-%b')
            
            def highlight_score(val):
                score = float(str(val).replace('%', ''))
                if score > 65: color = '#3fb950'
                elif score > 45: color = '#d29922'
                else: color = '#f85149'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                display_df.style.map(highlight_score, subset=['ML_Score'])
                .format({
                    'ML_Score': '{:.1f}%', 
                    'Entry': '₹{:.2f}', 
                    'Qty': '{:,.0f}',
                    'Investment': '₹{:,.0f}',
                    'Trailing_Stop': '₹{:.2f}', 
                    'Smart_Money_Score': '{:.0f}'
                }),
                use_container_width=True,
                height=400
            )
    else:
        st.info("No high-conviction signals in the selected lookback period.")

with tab2:
    st.markdown("### ⚙️ System Configuration")
    if st.button("🔄 Sync Data & Re-train AI (Complete Refresh)"):
        with st.spinner("Processing 5 Years of Data..."):
            # Call pipeline using the current python interpreter
            import sys
            subprocess.run([sys.executable, "pipeline.py"])
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    st.markdown("#### Model Features Influence")
    if metrics and metrics.get('walk_forward_scores'):
        st.json(metrics['walk_forward_scores'])

