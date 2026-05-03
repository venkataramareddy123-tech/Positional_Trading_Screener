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

st.set_page_config(layout="wide", page_title="Institutional Fingerprint Screener", page_icon="🏦")

st.markdown("""
<style>
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
    .metric-container {
        background: linear-gradient(145deg, rgba(22, 27, 34, 0.8), rgba(13, 17, 23, 0.9));
        backdrop-filter: blur(12px);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: transparent;
        border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(88, 166, 255, 0.1);
        border-bottom: 2px solid #58a6ff !important; color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_data.parquet")
MODEL_FILE = os.path.join(DATA_DIR, "xgb_model.pkl")
METRICS_FILE = os.path.join(DATA_DIR, "ml_metrics.json")
LABEL_DIAGNOSTICS_FILE = os.path.join(DATA_DIR, "label_diagnostics.json")
BASE_MODEL_FEATURES = [
    'Vol_Surge', 'Del_Surge', 'Close_Location',
    'RS_1D', 'RS_5D', 'RS_20D', 'ATR_30', 'BB_Width',
    'Nifty_vs_EMA50', 'Nifty_vs_EMA200', 'Dist_From_52W_High',
    'Smart_Money_Score'
]
VIX_MODEL_FEATURES = ['India_VIX', 'VIX_Change_5D', 'VIX_Rank_252D']
MODEL_FEATURES = BASE_MODEL_FEATURES + VIX_MODEL_FEATURES

@st.cache_data
def load_data():
    if not os.path.exists(PROCESSED_FILE) or not os.path.exists(MODEL_FILE):
        return None, None, None
    df = pd.read_parquet(PROCESSED_FILE)
    if 'Smart_Money_Score' not in df.columns:
        df['Smart_Money_Score'] = 0.0
    if 'Smart_Money_Hit' not in df.columns:
        df['Smart_Money_Hit'] = False
        
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
        
    metrics = None
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
            
    return df, model, metrics
    
def safe_predict_proba(model, X):
    try:
        return (model.predict_proba(X)[:, 1] * 100).round(2)
    except ValueError:
        return np.zeros(len(X))

def get_model_features(metrics, model):
    if metrics and metrics.get('features'):
        return metrics['features']
    expected_features = getattr(model, 'n_features_in_', None)
    if expected_features == len(MODEL_FEATURES):
        return MODEL_FEATURES
    return BASE_MODEL_FEATURES

def load_label_diagnostics():
    if os.path.exists(LABEL_DIAGNOSTICS_FILE):
        with open(LABEL_DIAGNOSTICS_FILE, 'r') as f:
            return json.load(f)
    return None

def run_pipeline_ui():
    progress_bar = st.progress(0)
    status_text = st.empty()
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    python_cmd = venv_python if os.path.exists(venv_python) else "python"
    try:
        process = subprocess.Popen([python_cmd, "pipeline.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(process.stdout.readline, ''):
            if not line: break
            if "PROGRESS:" in line:
                try:
                    parts = line.strip().split("PROGRESS:")[1].split("/")
                    current, total = int(parts[0]), int(parts[1])
                    if total > 0:
                        progress_bar.progress(min(current / total, 1.0))
                except: pass
            else:
                status_text.text(line.strip())
        process.wait()
        if process.returncode == 0:
            load_data.clear()
            status_text.success("Successfully completed!")
            time.sleep(1)
            st.rerun()
        else:
            status_text.error("Pipeline failed. Check logs.")
    except Exception as e:
        status_text.error(f"Error starting pipeline: {str(e)}")

df, model, metrics = load_data()

if df is None:
    st.warning("Data not found. Please use the Data Management section below to sync.")
    st.markdown("### ⚙️ Data Management")
    if st.button("🔄 Sync & Update Data (First Time)", width="stretch"):
        with st.spinner("Running Data Pipeline... This may take a while."):
            import subprocess
            try:
                venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
                python_cmd = venv_python if os.path.exists(venv_python) else "python"
                result = subprocess.run([python_cmd, "pipeline.py"], capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    load_data.clear()
                    st.success("Data successfully synced and updated!")
                    st.rerun()
                else:
                    st.error(f"Pipeline failed: {result.stderr}")
            except Exception as e:
                st.error(f"Error running pipeline: {str(e)}")
    st.stop()

model_features = get_model_features(metrics, model)
if any(feature not in df.columns for feature in model_features):
    model_features = BASE_MODEL_FEATURES
latest_date = df['Date'].max()
nifty_latest = df[df['Date'] == latest_date][['Nifty_Close', 'Nifty_EMA50', 'Nifty_EMA200', 'India_VIX']].iloc[0]

if nifty_latest['Nifty_Close'] > nifty_latest['Nifty_EMA50'] and nifty_latest['Nifty_EMA50'] > nifty_latest['Nifty_EMA200']:
    trend = "🟢 Strong Uptrend"
elif nifty_latest['Nifty_Close'] < nifty_latest['Nifty_EMA50'] and nifty_latest['Nifty_EMA50'] < nifty_latest['Nifty_EMA200']:
    trend = "🔴 Strong Downtrend"
elif nifty_latest['Nifty_Close'] > nifty_latest['Nifty_EMA50']:
    trend = "🟡 Weak Uptrend / Consolidation"
else:
    trend = "🟠 Weak Downtrend / Correction"

st.markdown('<div class="metric-container">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Market Status (Nifty 50)", trend)
with col2:
    vix_val = f"{nifty_latest['India_VIX']:.2f}" if pd.notna(nifty_latest['India_VIX']) else "N/A"
    st.metric("India VIX", vix_val)
with col3:
    st.metric("Nifty 50 Close", f"₹{nifty_latest['Nifty_Close']:.2f}")
with col4:
    total_hits = len(df[(df['Date'] == latest_date) & (df['Screener_Hit'] == True)])
    st.metric("Institutional Fingerprints Today", total_hits)
st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2, tab4, tab5, tab3 = st.tabs(["🚀 Live Screener & Predictions", "📊 Historical Analytics", "🔎 Stock Search & Patterns", "🤖 ML Model Evaluation", "⚙️ Data Management"])

with tab1:
    st.markdown("### 📡 Active Tracker & Live Screener")
    
    timeframe_opts_t1 = ["Today", "Last 3 Days", "Last 1 Week", "Last 15 Days", "Last 1 Month", "Last 2 Months", "Last 3 Months"]
    selected_tf_t1 = st.selectbox("Active Tracker Timeframe", timeframe_opts_t1, index=0)
    
    if selected_tf_t1 == "Today":
        start_dt_t1 = latest_date
    elif selected_tf_t1 == "Last 3 Days":
        start_dt_t1 = latest_date - pd.Timedelta(days=3)
    elif selected_tf_t1 == "Last 1 Week":
        start_dt_t1 = latest_date - pd.Timedelta(days=7)
    elif selected_tf_t1 == "Last 15 Days":
        start_dt_t1 = latest_date - pd.Timedelta(days=15)
    elif selected_tf_t1 == "Last 1 Month":
        start_dt_t1 = latest_date - pd.Timedelta(days=30)
    elif selected_tf_t1 == "Last 2 Months":
        start_dt_t1 = latest_date - pd.Timedelta(days=60)
    else:
        start_dt_t1 = latest_date - pd.Timedelta(days=90)
        
    st.markdown(f"*(Showing alerts from **{start_dt_t1.strftime('%b %d, %Y')}** to **{latest_date.strftime('%b %d, %Y')}**)*")
    
    latest_prices = df[df['Date'] == latest_date].set_index('Symbol')['Close']
    
    col_l1, col_l2 = st.columns(2)
    
    # Left Column: The Action List
    with col_l1:
        st.markdown("#### 🚀 The Action List (Breakouts)")
        st.caption("Massive volume surges.")
        action_hits = df[(df['Date'] >= start_dt_t1) & (df['Date'] <= latest_date) & (df['Screener_Hit'] == True)].copy()
        
        if action_hits.empty:
            st.info("No explosive breakouts detected in this timeframe.")
        else:
            action_hits = action_hits.sort_values('Date', ascending=False)
            # Dedup by keeping the most recent hit per symbol within the timeframe
            action_hits = action_hits.drop_duplicates(subset=['Symbol'], keep='first')
            
            X_live = action_hits[model_features].fillna(0)
            action_hits['ML Score (%)'] = safe_predict_proba(model, X_live)
            
            action_hits['Current_Price'] = action_hits['Symbol'].map(latest_prices)
            # Entry Price is the next day's open. If NaN (e.g. today), fall back to today's Close
            action_hits['Entry'] = action_hits['Entry_Price'].fillna(action_hits['Close'])
            action_hits['Live_PnL (%)'] = ((action_hits['Current_Price'] - action_hits['Entry']) / action_hits['Entry']) * 100
            
            display_cols = ['Date', 'Symbol', 'Entry', 'Current_Price', 'Live_PnL (%)', 'ML Score (%)']
            display_df = action_hits[display_cols].reset_index(drop=True)
            
            # Format Date
            display_df['Date'] = display_df['Date'].dt.strftime('%d-%b')
            
            def highlight_pnl(val):
                color = '#3fb950' if val > 0 else '#f85149' if val < 0 else '#c9d1d9'
                return f'color: {color}; font-weight: bold'
                
            def highlight_score(val):
                color = '#3fb950' if val > 65 else '#d29922' if val > 45 else '#f85149'
                return f'color: {color}; font-weight: bold'
                
            styled_df = display_df.style.map(highlight_score, subset=['ML Score (%)']) \
                                        .map(highlight_pnl, subset=['Live_PnL (%)']) \
                                        .format({'Entry': '₹{:.2f}', 'Current_Price': '₹{:.2f}', 'Live_PnL (%)': '{:+.2f}%', 'ML Score (%)': '{:.2f}%'})
                                        
            st.dataframe(
                styled_df, 
                width="stretch", 
                height=400,
                column_config={
                    "Entry": st.column_config.NumberColumn("Entry", help="Realistic execution price (assumes you buy at the Open on the morning after the alert)."),
                    "Current_Price": st.column_config.NumberColumn("Current Price", help="The most recent closing price of the stock."),
                    "Live_PnL (%)": st.column_config.TextColumn("Live PnL (%)", help="The live, running profit or loss since the alert triggered."),
                    "ML Score (%)": st.column_config.TextColumn("ML Score (%)", help="The AI's probability score that this breakout will generate >5% profit within 10 days.")
                }
            )
            
    # Right Column: The Stealth Watchlist
    with col_l2:
        st.markdown("#### 🤫 The Stealth Watchlist (Smart Money)")
        st.caption("Quiet accumulation detected.")
        stealth_hits = df[(df['Date'] >= start_dt_t1) & (df['Date'] <= latest_date) & (df['Smart_Money_Hit'] == True)].copy()
        
        if stealth_hits.empty:
            st.info("No stealth accumulation detected in this timeframe.")
        else:
            stealth_hits = stealth_hits.sort_values('Date', ascending=False)
            stealth_hits = stealth_hits.drop_duplicates(subset=['Symbol'], keep='first')
            
            stealth_hits['Current_Price'] = stealth_hits['Symbol'].map(latest_prices)
            stealth_hits['Entry'] = stealth_hits['Entry_Price'].fillna(stealth_hits['Close'])
            stealth_hits['Live_PnL (%)'] = ((stealth_hits['Current_Price'] - stealth_hits['Entry']) / stealth_hits['Entry']) * 100
            
            stealth_cols = ['Date', 'Symbol', 'Smart_Money_Score', 'Entry', 'Current_Price', 'Live_PnL (%)']
            stealth_df = stealth_hits[stealth_cols].reset_index(drop=True)
            stealth_df['Date'] = stealth_df['Date'].dt.strftime('%d-%b')
            
            def highlight_sm(val):
                return 'color: #8a2be2; font-weight: bold' if val > 90 else ''
                
            styled_sm = stealth_df.style.map(highlight_sm, subset=['Smart_Money_Score']) \
                                        .map(highlight_pnl, subset=['Live_PnL (%)']) \
                                        .format({'Entry': '₹{:.2f}', 'Current_Price': '₹{:.2f}', 'Live_PnL (%)': '{:+.2f}%', 'Smart_Money_Score': '{:.0f}/100'})
                                        
            st.dataframe(
                styled_sm, 
                width="stretch", 
                height=400,
                column_config={
                    "Smart_Money_Score": st.column_config.TextColumn("Smart Money Score", help="A 0-100 institutional accumulation score based on Delivery Surges, Volatility Contraction, and Up/Down Volume trends."),
                    "Entry": st.column_config.NumberColumn("Entry", help="Execution price if you had bought the morning after this stealth alert."),
                    "Current_Price": st.column_config.NumberColumn("Current Price", help="The most recent closing price of the stock."),
                    "Live_PnL (%)": st.column_config.TextColumn("Live PnL (%)", help="The live, running profit or loss since the Stealth Alert was detected.")
                }
            )

with tab2:
    st.subheader("What Worked in the Past?")
    
    latest_avail = latest_date - pd.Timedelta(days=15)
    min_date_avail = df['Date'].min().date() if not df.empty else datetime.today().date()
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        timeframe_opts = ["Last 1 Month", "Last 3 Months", "Last 6 Months", "Last 1 Year", "All Time", "Custom"]
        selected_tf = st.selectbox("Analysis Timeframe", timeframe_opts, index=1)
        
    if selected_tf == "Last 1 Month":
        start_dt = latest_avail - pd.Timedelta(days=30)
        end_dt = latest_avail
    elif selected_tf == "Last 3 Months":
        start_dt = latest_avail - pd.Timedelta(days=90)
        end_dt = latest_avail
    elif selected_tf == "Last 6 Months":
        start_dt = latest_avail - pd.Timedelta(days=180)
        end_dt = latest_avail
    elif selected_tf == "Last 1 Year":
        start_dt = latest_avail - pd.Timedelta(days=365)
        end_dt = latest_avail
    elif selected_tf == "All Time":
        start_dt = pd.to_datetime(min_date_avail)
        end_dt = latest_avail
    else:
        with col_t2:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                start_dt = pd.to_datetime(st.date_input("Start Date", latest_avail - pd.Timedelta(days=90), min_value=min_date_avail, max_value=latest_avail.date()))
            with col_c2:
                end_dt = pd.to_datetime(st.date_input("End Date", latest_avail.date(), min_value=min_date_avail, max_value=latest_avail.date()))
                
    past_hits = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt) & (df['Screener_Hit'] == True)].copy()
    
    if past_hits.empty:
        st.info("Not enough historical data to generate analytics for this timeframe.")
    else:
        st.markdown(f"### 🔍 The Playbook ({start_dt.strftime('%b %Y')} - {end_dt.strftime('%b %Y')})")
        
        del_3x = past_hits[past_hits['Del_Surge'] > 3.0]
        del_3x_win = del_3x['Target_Success'].mean() * 100 if not del_3x.empty else 0
        
        down_market_del = past_hits[(past_hits['Nifty_vs_EMA50'] < 0) & (past_hits['Del_Surge'] > 2.5)]
        down_market_win = down_market_del['Target_Success'].mean() * 100 if not down_market_del.empty else 0
        
        rs_breakouts = past_hits[past_hits['RS_20D'] > 0.05]
        rs_win = rs_breakouts['Target_Success'].mean() * 100 if not rs_breakouts.empty else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**>3x Delivery Surge**")
            st.markdown(f"<h2 style='color:#3fb950'>{del_3x_win:.1f}% Win Rate</h2>", unsafe_allow_html=True)
            avg_prof = del_3x['Max_Return_10D'].mean() * 100 if not del_3x.empty else 0
            st.caption(f"Avg Profit: {avg_prof:.1f}% | {len(del_3x)} setups")
        with col2:
            st.markdown(f"**Down Market + >2.5x Del**")
            st.markdown(f"<h2 style='color:#58a6ff'>{down_market_win:.1f}% Win Rate</h2>", unsafe_allow_html=True)
            avg_prof2 = down_market_del['Max_Return_10D'].mean() * 100 if not down_market_del.empty else 0
            st.caption(f"Avg Profit: {avg_prof2:.1f}% | {len(down_market_del)} setups")
        with col3:
            st.markdown(f"**High Relative Strength (+5% vs Nifty)**")
            st.markdown(f"<h2 style='color:#d29922'>{rs_win:.1f}% Win Rate</h2>", unsafe_allow_html=True)
            avg_prof3 = rs_breakouts['Max_Return_10D'].mean() * 100 if not rs_breakouts.empty else 0
            st.caption(f"Avg Profit: {avg_prof3:.1f}% | {len(rs_breakouts)} setups")
            
        st.markdown("---")
        st.markdown("### Advanced ML Probability Filter & Backtesting")
        
        X_past = past_hits[model_features].fillna(0)
        past_hits['ML Score (%)'] = safe_predict_proba(model, X_past)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            ml_threshold = st.slider("Min ML Score (%)", min_value=0, max_value=100, value=50, step=5)
        with col_m2:
            holding_period = st.slider("Target Holding Period (Days)", min_value=1, max_value=30, value=10, step=1)
        with col_m3:
            market_regime = st.selectbox("Market Regime", ["All Markets", "Bull Market (Nifty > 50EMA)", "Bear Market (Nifty < 50EMA)"])
            
        filtered_hits = past_hits[past_hits['ML Score (%)'] >= ml_threshold]
        
        if market_regime == "Bull Market (Nifty > 50EMA)":
            filtered_hits = filtered_hits[filtered_hits['Nifty_vs_EMA50'] > 0]
        elif market_regime == "Bear Market (Nifty < 50EMA)":
            filtered_hits = filtered_hits[filtered_hits['Nifty_vs_EMA50'] < 0]
            
        if filtered_hits.empty:
            st.info("No historical hits match these filters in the selected timeframe.")
        else:
            st.markdown(f"**Overview for Score >= {ml_threshold}% | {market_regime}**")
            
            # 1. Holding Period Breakdown
            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            with col_h1:
                st.metric("Total Setups", len(filtered_hits))
                if f'Max_Return_{holding_period}D' in filtered_hits.columns:
                    dynamic_win = (filtered_hits[f'Max_Return_{holding_period}D'] > 0.05).mean() * 100
                else:
                    dynamic_win = 0
                st.metric(f"Win Rate (T+{holding_period} > 5%)", f"{dynamic_win:.1f}%")
            with col_h2:
                prof_cust = filtered_hits[f'Max_Return_{holding_period}D'].mean() * 100 if f'Max_Return_{holding_period}D' in filtered_hits.columns else 0
                st.metric(f"Avg Profit (T+{holding_period})", f"{prof_cust:.1f}%")
            with col_h3:
                # Risk vs Reward (MAE)
                avg_drawdown = filtered_hits[f'Max_Drawdown_{holding_period}D'].mean() * 100 if f'Max_Drawdown_{holding_period}D' in filtered_hits.columns else 0
                st.metric(f"Avg Drawdown (T+{holding_period})", f"{avg_drawdown:.1f}%")
            with col_h4:
                profit_factor = abs(prof_cust / avg_drawdown) if avg_drawdown != 0 else 0
                st.metric(f"Profit Factor", f"{profit_factor:.2f}")

            # 2. Hit Rate Leaderboard
            st.markdown("### 🏆 Symbol Hit Rate Leaderboard")
            if f'Max_Return_{holding_period}D' in filtered_hits.columns:
                leaderboard = filtered_hits.groupby('Symbol').agg(
                    Total_Setups=('Date', 'count'),
                    Win_Rate=(f'Max_Return_{holding_period}D', lambda x: (x > 0.05).mean() * 100),
                    Avg_Profit=(f'Max_Return_{holding_period}D', lambda x: x.mean() * 100),
                    Max_Profit=(f'Max_Return_{holding_period}D', lambda x: x.max() * 100),
                    Avg_Drawdown=(f'Max_Drawdown_{holding_period}D', lambda x: x.mean() * 100)
                ).reset_index()
                
                # Calculate Profit Factor safely
                leaderboard['Profit_Factor'] = abs(leaderboard['Avg_Profit'] / leaderboard['Avg_Drawdown'].replace(0, -0.01))
                
                leaderboard = leaderboard[leaderboard['Total_Setups'] >= 2] # At least 2 trades to qualify
                leaderboard = leaderboard.sort_values('Profit_Factor', ascending=False)
                
                if not leaderboard.empty:
                    st.dataframe(leaderboard.style.format({
                        'Win_Rate': '{:.1f}%',
                        'Avg_Profit': '{:.1f}%',
                        'Max_Profit': '{:.1f}%',
                        'Avg_Drawdown': '{:.1f}%',
                        'Profit_Factor': '{:.2f}'
                    }), width="stretch")
                else:
                    st.info("Not enough repeated trades per symbol to generate a leaderboard.")
                    
            # 3. Cumulative Equity Curve
            st.markdown(f"### 📈 Cumulative Equity Curve (T+{holding_period} Holds)")
            st.caption(f"Cumulative Maximum Potential Return (%) based on taking every trade with T+{holding_period} holds.")
            
            eq_df = filtered_hits.sort_values('Date').copy()
            if f'Max_Return_{holding_period}D' in eq_df.columns:
                eq_df['Cumulative_Return (%)'] = eq_df[f'Max_Return_{holding_period}D'].cumsum() * 100
                if not eq_df.empty:
                    st.line_chart(eq_df.set_index('Date')['Cumulative_Return (%)'], height=300)

with tab4:
    st.subheader("Stock Deep Dive & Pattern Search")
    
    all_symbols = sorted(df['Symbol'].unique())
    selected_symbol = st.selectbox("Select Stock Symbol", all_symbols)
    
    stock_hits = df[(df['Symbol'] == selected_symbol) & (df['Screener_Hit'] == True)].copy()
    
    if stock_hits.empty:
        st.info(f"No historical screener hits found for {selected_symbol}.")
    else:
        st.write(f"### Historical Screener Hits for {selected_symbol}")
        display_cols = ['Date', 'Close', 'Vol_Surge', 'Del_Surge', 'RS_20D', 'Max_Return_10D', 'Target_Success']
        
        # Format the display dataframe
        st_stock_hits = stock_hits[display_cols].sort_values('Date', ascending=False).copy()
        st_stock_hits['Date'] = st_stock_hits['Date'].dt.strftime('%Y-%m-%d')
        
        styled_stock_hits = st_stock_hits.style.format({
            'Close': '₹{:.2f}',
            'Vol_Surge': '{:.1f}x',
            'Del_Surge': '{:.1f}x',
            'RS_20D': '{:.2%}',
            'Max_Return_10D': '{:.2%}'
        })
        st.dataframe(styled_stock_hits, width="stretch")
        
        st.write("---")
        st.write("### Find Similar Patterns Market-Wide")
        selected_date = st.selectbox("Select a past hit date to find similar patterns", st_stock_hits['Date'].tolist())
        
        if selected_date:
            target_record = stock_hits[stock_hits['Date'].dt.strftime('%Y-%m-%d') == selected_date].iloc[0]
            
            features_for_sim = ['Vol_Surge', 'Del_Surge', 'RS_1D', 'RS_5D', 'RS_20D']
            all_hits = df[df['Screener_Hit'] == True].copy()
            
            sim_df = all_hits.copy()
            for col in features_for_sim:
                mean_val = sim_df[col].mean()
                std_val = sim_df[col].std()
                if std_val > 0:
                    sim_df[f'{col}_norm'] = (sim_df[col] - mean_val) / std_val
                    target_val_norm = (target_record[col] - mean_val) / std_val
                else:
                    sim_df[f'{col}_norm'] = 0
                    target_val_norm = 0
                    
                sim_df[f'{col}_dist'] = (sim_df[f'{col}_norm'] - target_val_norm) ** 2
                
            sim_df['Total_Dist'] = sim_df[[f'{col}_dist' for col in features_for_sim]].sum(axis=1)
            sim_df = sim_df.sort_values('Total_Dist')
            
            # exclude exact match
            similar_setups = sim_df[~((sim_df['Symbol'] == target_record['Symbol']) & (sim_df['Date'] == target_record['Date']))].head(5)
            
            st.write(f"**Top 5 Similar Historical Setups to {selected_symbol} on {selected_date}**")
            st_sim = similar_setups[['Symbol', 'Date', 'Close', 'Vol_Surge', 'Del_Surge', 'RS_20D', 'Max_Return_10D', 'Target_Success']].copy()
            st_sim['Date'] = st_sim['Date'].dt.strftime('%Y-%m-%d')
            
            styled_sim = st_sim.style.format({
                'Close': '₹{:.2f}',
                'Vol_Surge': '{:.1f}x',
                'Del_Surge': '{:.1f}x',
                'RS_20D': '{:.2%}',
                'Max_Return_10D': '{:.2%}'
            })
            st.dataframe(styled_sim, width="stretch")

with tab5:
    st.subheader("Machine Learning Model Evaluation")
    label_diagnostics = load_label_diagnostics()
    if label_diagnostics is not None:
        st.markdown("### Training Label Quality")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Usable Labels", f"{label_diagnostics.get('usable_training_labels', 0):,}")
        with c2:
            st.metric("Label Completeness", f"{label_diagnostics.get('label_completeness', 0):.2%}")
        with c3:
            st.metric("Positive Label Rate", f"{label_diagnostics.get('positive_label_rate', 0):.2%}")
        with c4:
            st.metric("Suspicious Return Labels", f"{label_diagnostics.get('suspicious_return_rate_gt_100pct', 0):.2%}")
    if metrics is not None:
        st.markdown("### Top Performing Models")
        top_models = metrics.get('top_models', [])
        st.success(f"**Current Ensemble:** {' & '.join(top_models) if top_models else 'Not available'}")
        
        st.markdown("### Walk-Forward Cross Validation Performance")
        selection_metric = metrics.get('selection_metric', 'average_precision')
        display_df = None
        metric_labels = {
            'average_precision': 'Average Precision',
            'roc_auc': 'ROC AUC',
            'f1': 'F1',
            'balanced_accuracy': 'Balanced Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'accuracy': 'Accuracy',
        }
        st.caption(f"Models are selected by {metric_labels.get(selection_metric, selection_metric)} because ML Score ranks breakout candidates by probability.")
        if 'walk_forward_metrics' in metrics and metrics['walk_forward_metrics']:
            score_df = pd.DataFrame.from_dict(metrics['walk_forward_metrics'], orient='index').reset_index(names='Algorithm')
            metric_cols = ['average_precision', 'roc_auc', 'f1', 'balanced_accuracy', 'precision', 'recall', 'accuracy']
            metric_cols = [col for col in metric_cols if col in score_df.columns]
            score_df = score_df[['Algorithm'] + metric_cols]
            score_df = score_df.sort_values(selection_metric, ascending=False)
            display_df = score_df.rename(columns=metric_labels)
            format_cols = {metric_labels.get(col, col): '{:.2%}' for col in metric_cols}
            st.dataframe(display_df.style.format(format_cols), width="stretch")
        else:
            scores = metrics.get('walk_forward_scores', {})
            score_df = pd.DataFrame(list(scores.items()), columns=['Algorithm', metric_labels.get(selection_metric, selection_metric)])
            score_df = score_df.sort_values(metric_labels.get(selection_metric, selection_metric), ascending=False)
            st.dataframe(score_df.style.format({metric_labels.get(selection_metric, selection_metric): '{:.2%}'}), width="stretch")
        
        chart_col = metric_labels.get(selection_metric, selection_metric)
        if display_df is not None and chart_col in display_df.columns:
            st.bar_chart(display_df.set_index('Algorithm')[[chart_col]], height=400)
        else:
            st.bar_chart(score_df.set_index('Algorithm'), height=400)
        
        st.markdown("---")
        st.markdown("### Top Feature Importances")
        
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'estimators_'):
            for est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importances = est.feature_importances_
                    break
                    
        if importances is not None:
            if len(importances) == len(model_features):
                feat_df = pd.DataFrame({
                    'Feature': model_features,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                st.bar_chart(feat_df.set_index('Feature'), height=400)
            else:
                st.info("Feature importances are unavailable because the saved model and feature list are out of sync. Run the pipeline again.")
        else:
            st.info("Feature importances not available for the current model ensemble.")

        st.markdown("---")
        if st.button("🔄 Force Re-evaluate Models", key="reeval_models"):
            run_pipeline_ui()

    else:
        st.info("No ML evaluation metrics found. Please run the ML Pipeline to evaluate algorithms and generate metrics.")
        if st.button("🚀 Run ML Evaluation Pipeline", key="run_ml_eval"):
            run_pipeline_ui()

with tab3:
    st.subheader("Data Management")
    st.markdown("Update daily data, sync with historical data, or delete existing datasets.")
    
    st.markdown("### Sync & Update")
    if st.button("🔄 Sync & Update Data", width="stretch"):
        run_pipeline_ui()
            
    st.markdown("### Delete Custom Timeframe Data")
    col_del1, col_del2 = st.columns(2)
    with col_del1:
        del_start = st.date_input("Delete From", value=datetime.today().date() - pd.Timedelta(days=30))
    with col_del2:
        del_end = st.date_input("Delete Until", value=datetime.today().date())
        
    if st.button("🗑️ Delete Timeframe Data", type="primary"):
        try:
            if os.path.exists(RAW_FILE):
                raw_df = pd.read_parquet(RAW_FILE)
                if 'Date' in raw_df.columns:
                    mask = (raw_df['Date'].dt.date >= del_start) & (raw_df['Date'].dt.date <= del_end)
                    raw_df_filtered = raw_df[~mask]
                    raw_df_filtered.to_parquet(RAW_FILE, index=False)
                    st.success(f"Deleted {mask.sum()} rows from raw data between {del_start} and {del_end}. Please run Sync to regenerate features.")
            if os.path.exists(PROCESSED_FILE): os.remove(PROCESSED_FILE)
            load_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting timeframe: {str(e)}")

    st.markdown("### Delete All Data")
    if st.button("🗑️ Delete All Data", type="primary"):
        try:
            if os.path.exists(RAW_FILE): os.remove(RAW_FILE)
            if os.path.exists(NIFTY_FILE): os.remove(NIFTY_FILE)
            if os.path.exists(PROCESSED_FILE): os.remove(PROCESSED_FILE)
            if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
            load_data.clear()
            st.success("All data deleted successfully. Please run sync to fetch again.")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting files: {str(e)}")
