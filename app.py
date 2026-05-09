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
    'Vol_Surge', 'Del_Surge', 'Close_Location',
    'RS_1D', 'RS_5D', 'RS_20D', 'ATR_30', 'BB_Width',
    'Nifty_vs_EMA50', 'Nifty_vs_EMA200', 'Dist_From_52W_High',
    'Smart_Money_Score', 'Is_Inst_Buy', 'Inst_Intensity', 
    'Del_Anomaly', 'Vol_Trend', 'Dist_From_EMA50',
    'India_VIX', 'VIX_Change_5D', 'VIX_Rank_252D'
]

@st.cache_data
def load_data(mtime):
    if not os.path.exists(PROCESSED_FILE) or not os.path.exists(MODEL_FILE):
        return None, None, None
    df = pd.read_parquet(PROCESSED_FILE)
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    metrics = None
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
    return df, model, metrics

def calculate_levels(row):
    if pd.notna(row['Entry_Price']):
        entry = row['Entry_Price']
    else:
        # Add a 1.0% gap buffer for a more realistic entry expectation on live signals
        entry = row['Close'] * 1.01

    # 2.0x ATR Stop - Alignment with pipeline simulation
    stop_dist = 2.0 * row['ATR_30']
    stop_pct = stop_dist / (entry + 1e-8)
    
    # Enforce min 3% stop and max 10% stop as per pipeline survival rules
    actual_stop_pct = max(0.03, min(0.10, stop_pct))
    stop_loss = entry * (1 - actual_stop_pct)
        
    # 15% Target (Home Run target)
    target = entry * 1.15
    return pd.Series([entry, target, stop_loss])

mtime = os.path.getmtime(PROCESSED_FILE) if os.path.exists(PROCESSED_FILE) else 0
df, model, metrics = load_data(mtime)

# --- HEADER ---
st.title("🏦 Institutional Edge Trader")
if df is not None:
    latest_date = df['Date'].max()
    st.caption(f"Last Updated: {latest_date.strftime('%d %b, %Y')} | Data: NSE Cash | Engine: XGBoost 2.0")
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
        quality_filter = st.selectbox("Signal Quality Filter", options=["All Signals", "Institutional Grade (Score > 65)", "High Conviction (Score > 50)", "Solid Setup (Score > 45)", "Standard (Score > 35)"], index=0)
    with col_f2:
        session_opts = {
            "Last Session": 1, 
            "Last 3 Sessions": 3, 
            "Last 5 Sessions": 5, 
            "Last 10 Sessions": 10, 
            "Last 20 Sessions": 20,
            "Last 30 Sessions (~1.5 Months)": 30,
            "Last 60 Sessions (~3 Months)": 60,
            "Last 120 Sessions (~6 Months)": 120,
            "Last 252 Sessions (~1 Year)": 252,
            "Last 504 Sessions (~2 Years)": 504,
            "Last 756 Sessions (~3 Years)": 756,
            "Last 1008 Sessions (~4 Years)": 1008,
            "Last 1260 Sessions (~5 Years)": 1260,
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
        signals[['Entry', 'Target', 'Stop_Loss']] = signals.apply(calculate_levels, axis=1)
        
        # Calculate Position Sizing
        # Qty = Risk_Amount / (Entry - Stop)
        signals['Qty'] = (risk_amt / (signals['Entry'] - signals['Stop_Loss'] + 1e-8)).astype(int)
        signals['Investment'] = signals['Qty'] * signals['Entry']
        
        # Predict ML Score
        X = signals[MODEL_FEATURES].fillna(0)
        signals['ML_Score'] = model.predict_proba(X)[:, 1] * 100
        
        # Apply Quality Filter using Fixed Absolute Tiers
        # These correspond to historical performance percentiles (90th, 75th, 50th)
        if quality_filter == "Institutional Grade (Score > 65)":
            signals = signals[signals['ML_Score'] >= 65]
        elif quality_filter == "High Conviction (Score > 50)":
            signals = signals[signals['ML_Score'] >= 50]
        elif quality_filter == "Solid Setup (Score > 45)":
            signals = signals[signals['ML_Score'] >= 45]
        elif quality_filter == "Standard (Score > 35)":
            signals = signals[signals['ML_Score'] >= 35]
            
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
                    actual_stop_pct = (1 - sig['Stop_Loss'] / sig['Entry']) * 100
                    
                    with row_cols[j]:
                        st.markdown(f"""
                        <div class="signal-card">
                            <div class="card-header">
                                <span class="symbol-name">{sig['Symbol']}</span>
                                <span class="signal-date">{sig['Date'].strftime('%d %b')}</span>
                            </div>
                            <div style="text-align: center; margin-bottom: 20px;">
                                <div class="metric-label">AI Confidence Score</div>
                                <div class="score-badge {score_class}">{score:.1f}%</div>
                            </div>
                            <div class="metric-grid">
                                <div class="metric-box">
                                    <div class="metric-label">Buy Price</div>
                                    <div class="metric-value">₹{sig['Entry']:,.2f}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">Quantity</div>
                                    <div class="metric-value">{int(sig['Qty'])}</div>
                                </div>
                            </div>
                            <div class="level-container">
                                <div class="level-row">
                                    <span>Target (+15%)</span>
                                    <span class="target-val">₹{sig['Target']:,.2f}</span>
                                </div>
                                <div class="level-row">
                                    <span>Stop Loss (-{actual_stop_pct:.1f}%)</span>
                                    <span class="stop-val">₹{sig['Stop_Loss']:,.2f}</span>
                                </div>
                                <div class="level-row" style="margin-top: 10px; border-top: 1px dashed #30363d; padding-top: 10px;">
                                    <span style="color: #8b949e;">Total Investment</span>
                                    <span style="color: #ffffff; font-weight: 700;">₹{sig['Investment']:,.0f}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with st.expander("🔍 View Raw Signal Data Grid", expanded=False):
            display_df = signals[['Date', 'Symbol', 'ML_Score', 'Entry', 'Qty', 'Investment', 'Target', 'Stop_Loss', 'Smart_Money_Score']].copy()
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
                    'Target': '₹{:.2f}', 
                    'Stop_Loss': '₹{:.2f}', 
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

