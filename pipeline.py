import os

# Keep data refresh/model training on CPU by default to avoid waking NVIDIA on battery.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import nselib
from nselib import capital_market
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ta
import pickle
import logging
import json
import warnings
import re

# Suppress pandas fragmentation warnings during the 30-day loop
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
RAW_FILE = os.path.join(DATA_DIR, "raw_data.parquet")
NIFTY_FILE = os.path.join(DATA_DIR, "nifty_data.parquet")
CORPORATE_ACTIONS_FILE = os.path.join(DATA_DIR, "corporate_actions.parquet")
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_data.parquet")
MODEL_FILE = os.path.join(DATA_DIR, "xgb_model.pkl")
LABEL_DIAGNOSTICS_FILE = os.path.join(DATA_DIR, "label_diagnostics.json")
SELECTION_METRIC = "average_precision"
MODEL_FEATURES = [
    'Vol_Surge', 'Del_Surge', 'Close_Location',
    'RS_1D', 'RS_5D', 'RS_20D', 'ATR_30', 'BB_Width',
    'Nifty_vs_EMA50', 'Nifty_vs_EMA200', 'Dist_From_52W_High',
    'Smart_Money_Score', 'India_VIX', 'VIX_Change_5D', 'VIX_Rank_252D'
]

def clean_numeric_series(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False).str.replace('-', '', regex=False), errors='coerce')

def load_cached_nifty_benchmark():
    if os.path.exists(NIFTY_FILE):
        logging.warning("Using existing local benchmark file.")
        return pd.read_parquet(NIFTY_FILE)
    return pd.DataFrame()

def standardize_nselib_benchmark(nifty, vix):
    if nifty.empty:
        return pd.DataFrame()

    nifty = nifty.copy()
    nifty['Date'] = pd.to_datetime(nifty['TIMESTAMP'], format='%d-%b-%Y', errors='coerce')
    nifty['Nifty_Close'] = clean_numeric_series(nifty['CLOSE_INDEX_VAL'])
    nifty = nifty[['Date', 'Nifty_Close']].dropna(subset=['Date', 'Nifty_Close'])
    nifty = nifty.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
    nifty['Nifty_EMA50'] = nifty['Nifty_Close'].ewm(span=50, adjust=False).mean()
    nifty['Nifty_EMA200'] = nifty['Nifty_Close'].ewm(span=200, adjust=False).mean()

    if not vix.empty:
        vix = vix.copy()
        vix['Date'] = pd.to_datetime(vix['TIMESTAMP'], format='%d-%b-%Y', errors='coerce')
        vix['India_VIX'] = clean_numeric_series(vix['CLOSE_INDEX_VAL'])
        vix = vix[['Date', 'India_VIX']].dropna(subset=['Date'])
        vix = vix.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
        nifty = pd.merge(nifty, vix, on='Date', how='left')
    else:
        nifty['India_VIX'] = np.nan

    return nifty[['Date', 'Nifty_Close', 'Nifty_EMA50', 'Nifty_EMA200', 'India_VIX']]

def fetch_nifty_benchmark_from_nselib(start_date, end_date):
    chunk_start = start_date
    nifty_parts = []
    vix_parts = []

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=89), end_date)
        from_date = chunk_start.strftime('%d-%m-%Y')
        to_date = chunk_end.strftime('%d-%m-%Y')

        nifty_chunk = capital_market.index_data(index='NIFTY 50', from_date=from_date, to_date=to_date)
        if not nifty_chunk.empty:
            nifty_parts.append(nifty_chunk)

        try:
            vix_chunk = capital_market.india_vix_data(from_date=from_date, to_date=to_date)
            if not vix_chunk.empty:
                vix_parts.append(vix_chunk)
        except Exception as e:
            logging.info(f"nselib India VIX unavailable for {from_date} to {to_date}; fallback will fill if needed. Detail: {e}")

        chunk_start = chunk_end + timedelta(days=1)

    nifty = pd.concat(nifty_parts, ignore_index=True) if nifty_parts else pd.DataFrame()
    vix = pd.concat(vix_parts, ignore_index=True) if vix_parts else pd.DataFrame()
    if vix.empty:
        logging.info("nselib India VIX unavailable for all chunks. Trying VIX fallback.")
    benchmark = standardize_nselib_benchmark(nifty, vix)
    if benchmark.empty:
        raise ValueError("nselib returned empty NIFTY benchmark data.")
    if benchmark['India_VIX'].isna().any():
        vix_fallback = fetch_vix_from_yahoo(start_date, end_date)
        if not vix_fallback.empty:
            benchmark = pd.merge(
                benchmark,
                vix_fallback.rename(columns={'India_VIX': 'India_VIX_Fallback'}),
                on='Date',
                how='left'
            )
            benchmark['India_VIX'] = benchmark['India_VIX'].combine_first(benchmark['India_VIX_Fallback'])
            benchmark = benchmark.drop(columns=['India_VIX_Fallback'])
    benchmark['India_VIX'] = benchmark['India_VIX'].ffill().bfill()
    benchmark.to_parquet(NIFTY_FILE, index=False)
    logging.info("Fetched Nifty 50 and India VIX benchmark data from nselib/NSE.")
    return benchmark

def fetch_vix_from_yahoo(start_date, end_date):
    vix = yf.download('^INDIAVIX', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    vix = vix.reset_index()
    if vix.empty:
        return pd.DataFrame()
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [col[0] if col[0] != 'Date' else 'Date' for col in vix.columns]
    vix = vix.rename(columns={'Close': 'India_VIX'})
    vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
    return vix[['Date', 'India_VIX']]

def get_benchmark_trading_dates(nifty_df, days_back):
    if nifty_df is None or nifty_df.empty or 'Date' not in nifty_df.columns:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back * 1.5)
        days = pd.date_range(start=start_date, end=end_date, freq='B')
        return pd.Series(days[-days_back:]).dt.normalize()
    dates = pd.to_datetime(nifty_df['Date']).dt.tz_localize(None).dt.normalize()
    return dates.dropna().drop_duplicates().sort_values().tail(days_back)

def fetch_historical_bhavcopies(days_back=1250, nifty_df=None):
    """Fetches Bhavcopy with delivery for the last N trading days."""
    trading_dates = get_benchmark_trading_dates(nifty_df, days_back)
    all_data = []
    
    if os.path.exists(RAW_FILE):
        logging.info("Loading existing raw data...")
        df_existing = pd.read_parquet(RAW_FILE)
        if 'Date' in df_existing.columns:
            existing_dates = pd.to_datetime(df_existing['Date']).dt.normalize()
            existing_date_set = set(existing_dates)
            latest_existing_date = existing_dates.max()
        else:
            existing_date_set = set()
            latest_existing_date = pd.NaT
    else:
        df_existing = pd.DataFrame()
        existing_date_set = set()
        latest_existing_date = pd.NaT

    backfill_missing_dates = os.environ.get('BACKFILL_MISSING_DATES', '0') == '1'
    if len(existing_date_set) == 0:
        dates_to_fetch = list(trading_dates)
    elif backfill_missing_dates:
        dates_to_fetch = [d for d in trading_dates if d not in existing_date_set]
    else:
        dates_to_fetch = [d for d in trading_dates if d > latest_existing_date]

    total_dates = len(dates_to_fetch)
    new_data_fetched = False
    if total_dates == 0:
        logging.info("Raw bhavcopy data is already in sync with the available NSE benchmark dates.")
    
    for i, trading_date in enumerate(dates_to_fetch):
        date_str = trading_date.strftime('%d-%m-%Y')
        logging.info(f"PROGRESS:{i}/{total_dates}")
        logging.info(f"Fetching data for {date_str}...")
        try:
            df_day = capital_market.bhav_copy_with_delivery(date_str)
            
            # Standardize column names
            df_day.columns = df_day.columns.str.strip().str.upper()
            
            col_map = {
                'SYMBOL': 'Symbol',
                'SERIES': 'Series',
                'OPEN_PRICE': 'Open',
                'HIGH_PRICE': 'High',
                'LOW_PRICE': 'Low',
                'CLOSE_PRICE': 'Close',
                'PREV_CLOSE': 'Prev_Close',
                'TTL_TRD_QNTY': 'Volume',
                'TURNOVER_LACS': 'Turnover',
                'DELIV_QTY': 'Delivery_Volume',
                'DELIV_PER': 'Delivery_Percent'
            }
            
            if 'SERIES' in df_day.columns:
                df_day = df_day[df_day['SERIES'] == 'EQ']
            
            df_day = df_day.rename(columns=col_map)
            df_day['Date'] = pd.to_datetime(date_str, format='%d-%m-%Y')
            
            req_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover', 'Delivery_Volume', 'Delivery_Percent']
            for col in req_cols:
                if col not in df_day.columns:
                    df_day[col] = np.nan
                    
            df_day = df_day[req_cols]
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover', 'Delivery_Volume', 'Delivery_Percent']:
                df_day[col] = clean_numeric_series(df_day[col])
                
            all_data.append(df_day)
            new_data_fetched = True
            
            # Save incrementally every 50 dates to prevent data loss on interruption
            if len(all_data) >= 50:
                df_new = pd.concat(all_data, ignore_index=True)
                df_existing = pd.concat([df_existing, df_new], ignore_index=True)
                df_existing = df_existing.drop_duplicates(subset=['Date', 'Symbol'])
                df_existing.to_parquet(RAW_FILE, index=False)
                all_data = [] # Reset buffer
                
            time.sleep(0.5)
        except Exception as e:
            message = str(e)
            if "Data not found" in message:
                logging.info(f"NSE has no bhavcopy delivery data for {date_str}; skipping.")
            else:
                logging.warning(f"Failed to fetch {date_str}: {e}")
            
    if total_dates > 0:
        logging.info(f"PROGRESS:{total_dates}/{total_dates}")
        
    if new_data_fetched and len(all_data) > 0:
        df_new = pd.concat(all_data, ignore_index=True)
        df_existing = pd.concat([df_existing, df_new], ignore_index=True)
        df_existing = df_existing.drop_duplicates(subset=['Date', 'Symbol'])
        df_existing.to_parquet(RAW_FILE, index=False)
        
    return df_existing

def fetch_nifty_benchmark(days_back=1250):
    logging.info("Fetching Nifty 50 and India VIX Benchmark Data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back * 1.5)

    try:
        return fetch_nifty_benchmark_from_nselib(start_date, end_date)
    except Exception as e:
        logging.warning(f"nselib benchmark fetch failed: {e}. Trying Yahoo Finance.")
    
    nifty = yf.download('^NSEI', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    nifty = nifty.reset_index()
    
    if nifty.empty:
        logging.warning("Failed to fetch Nifty data from Yahoo Finance.")
        return load_cached_nifty_benchmark()
        
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [col[0] if col[0] != 'Date' else 'Date' for col in nifty.columns]
        
    nifty = nifty.rename(columns={'Close': 'Nifty_Close'})
    nifty['Date'] = pd.to_datetime(nifty['Date']).dt.tz_localize(None)
    
    nifty['Nifty_EMA50'] = nifty['Nifty_Close'].ewm(span=50, adjust=False).mean()
    nifty['Nifty_EMA200'] = nifty['Nifty_Close'].ewm(span=200, adjust=False).mean()
    
    vix = yf.download('^INDIAVIX', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    vix = vix.reset_index()
    if not vix.empty:
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [col[0] if col[0] != 'Date' else 'Date' for col in vix.columns]
        vix = vix.rename(columns={'Close': 'India_VIX'})
        vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
        nifty = pd.merge(nifty, vix[['Date', 'India_VIX']], on='Date', how='left')
    else:
        nifty['India_VIX'] = np.nan

    nifty = nifty.sort_values('Date')
    nifty['India_VIX'] = nifty['India_VIX'].ffill().bfill()
    nifty = nifty[['Date', 'Nifty_Close', 'Nifty_EMA50', 'Nifty_EMA200', 'India_VIX']]
    nifty.to_parquet(NIFTY_FILE, index=False)
    logging.info("Fetched Nifty 50 and India VIX benchmark data from Yahoo Finance.")
    return nifty

def normalize_corporate_actions_columns(actions):
    rename_map = {}
    for col in actions.columns:
        normalized = col.replace('-', '').replace('_', '').lower()
        if normalized == 'symbol':
            rename_map[col] = 'Symbol'
        elif normalized in ['subject', 'purpose']:
            rename_map[col] = 'Purpose'
        elif normalized == 'exdate':
            rename_map[col] = 'Ex-Date'
        elif normalized == 'comp':
            rename_map[col] = 'Company'
        elif normalized == 'isin':
            rename_map[col] = 'ISIN'
    return actions.rename(columns=rename_map)

def parse_corporate_action_factor(purpose):
    text = str(purpose)
    split_match = re.search(r'From\s+Rs\.?\s*(\d+(?:\.\d+)?)\s*/?-?\s*.*?\s+To\s+R(?:s|e)\.?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if split_match:
        old_face = float(split_match.group(1))
        new_face = float(split_match.group(2))
        if old_face > 0 and new_face > 0:
            return old_face / new_face

    bonus_match = re.search(r'\bBonus\s+(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if bonus_match:
        bonus_shares = float(bonus_match.group(1))
        existing_shares = float(bonus_match.group(2))
        if existing_shares > 0:
            return (bonus_shares + existing_shares) / existing_shares

    return np.nan

def fetch_corporate_actions(days_back=1250):
    logging.info("Fetching NSE Corporate Actions for splits and bonuses...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back * 1.5)
    chunk_start = start_date
    parts = []

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=364), end_date)
        from_date = chunk_start.strftime('%d-%m-%Y')
        to_date = chunk_end.strftime('%d-%m-%Y')
        try:
            actions_chunk = capital_market.corporate_actions_for_equity(from_date=from_date, to_date=to_date)
            if not actions_chunk.empty:
                parts.append(actions_chunk)
        except Exception as e:
            logging.warning(f"Corporate actions fetch failed for {from_date} to {to_date}: {e}")
        chunk_start = chunk_end + timedelta(days=1)

    if parts:
        actions = pd.concat(parts, ignore_index=True)
        actions = normalize_corporate_actions_columns(actions)
        if {'Symbol', 'Purpose', 'Ex-Date'}.issubset(actions.columns):
            actions['Ex-Date'] = pd.to_datetime(actions['Ex-Date'], errors='coerce')
            actions['Adjustment_Factor'] = actions['Purpose'].apply(parse_corporate_action_factor)
            actions = actions.dropna(subset=['Symbol', 'Ex-Date', 'Adjustment_Factor'])
            actions = actions[actions['Adjustment_Factor'] > 0]
            actions = actions.drop_duplicates(subset=['Symbol', 'Ex-Date', 'Purpose'])
            actions.to_parquet(CORPORATE_ACTIONS_FILE, index=False)
            logging.info(f"Loaded {len(actions)} split/bonus corporate actions from NSE.")
            return actions

    if os.path.exists(CORPORATE_ACTIONS_FILE):
        logging.warning("Using existing local corporate actions file.")
        actions = pd.read_parquet(CORPORATE_ACTIONS_FILE)
        actions = normalize_corporate_actions_columns(actions)
        if 'Adjustment_Factor' not in actions.columns and 'Purpose' in actions.columns:
            actions['Adjustment_Factor'] = actions['Purpose'].apply(parse_corporate_action_factor)
        actions['Ex-Date'] = pd.to_datetime(actions['Ex-Date'], errors='coerce')
        return actions.dropna(subset=['Symbol', 'Ex-Date', 'Adjustment_Factor'])

    logging.warning("No corporate actions data available. Prices will not be split/bonus adjusted.")
    return pd.DataFrame(columns=['Symbol', 'Ex-Date', 'Purpose', 'Adjustment_Factor'])

def adjust_group_with_corporate_actions(df, actions):
    df = df.sort_values('Date').reset_index(drop=True)
    df['Cum_Adj'] = 1.0

    if not actions.empty:
        symbol_actions = actions.sort_values('Ex-Date')
        for _, action in symbol_actions.iterrows():
            factor = action['Adjustment_Factor']
            ex_date = action['Ex-Date']
            if pd.notna(factor) and factor > 0 and pd.notna(ex_date):
                df.loc[df['Date'] < ex_date, 'Cum_Adj'] *= factor

    df['Open'] = df['Open'] / df['Cum_Adj']
    df['High'] = df['High'] / df['Cum_Adj']
    df['Low'] = df['Low'] / df['Cum_Adj']
    df['Close'] = df['Close'] / df['Cum_Adj']
    df['Volume'] = df['Volume'] * df['Cum_Adj']
    df['Delivery_Volume'] = df['Delivery_Volume'] * df['Cum_Adj']

    return df.drop(columns=['Cum_Adj'])

def apply_corporate_action_adjustments(df, corporate_actions):
    logging.info("Applying NSE Corporate Action Adjustments (Splits/Bonuses)...")
    corporate_actions = normalize_corporate_actions_columns(corporate_actions.copy())
    if not corporate_actions.empty:
        corporate_actions['Ex-Date'] = pd.to_datetime(corporate_actions['Ex-Date'], errors='coerce')
        if 'Adjustment_Factor' not in corporate_actions.columns:
            corporate_actions['Adjustment_Factor'] = corporate_actions['Purpose'].apply(parse_corporate_action_factor)
        corporate_actions = corporate_actions.dropna(subset=['Symbol', 'Ex-Date', 'Adjustment_Factor'])
        corporate_actions = corporate_actions[corporate_actions['Adjustment_Factor'] > 0]
    actions_by_symbol = {symbol: group for symbol, group in corporate_actions.groupby('Symbol')}

    adjusted_df = (
        df.groupby('Symbol', group_keys=True)
        .apply(lambda g: adjust_group_with_corporate_actions(g, actions_by_symbol.get(g.name, pd.DataFrame())))
        .reset_index(level=0)
    )
    return adjusted_df.reset_index(drop=True)

def engineer_features(df, nifty_df):
    logging.info("Engineering Features and Screener Logic...")
    if nifty_df.empty:
        raise ValueError("Nifty benchmark data is empty; run the pipeline with network access or restore data/nifty_data.parquet.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
    benchmark_cols = ['Nifty_Close', 'Nifty_EMA50', 'Nifty_EMA200', 'India_VIX']
    missing_benchmark_cols = [col for col in benchmark_cols if col not in nifty_df.columns]
    if missing_benchmark_cols:
        raise ValueError(f"Missing benchmark columns: {missing_benchmark_cols}")
    if nifty_df[['Nifty_Close', 'Nifty_EMA50', 'Nifty_EMA200']].isna().any().any():
        raise ValueError("Nifty benchmark data contains missing values; cannot calculate relative strength reliably.")
    if nifty_df['India_VIX'].isna().all():
        logging.warning("India VIX data is unavailable. VIX features will be neutralized.")
        nifty_df['India_VIX'] = 0.0
    else:
        nifty_df['India_VIX'] = nifty_df['India_VIX'].ffill().bfill()
    
    df = pd.merge(df, nifty_df, on='Date', how='left').sort_values(['Date', 'Symbol'])
    missing_nifty_ratio = df['Nifty_Close'].isna().mean()
    if missing_nifty_ratio > 0.05:
        raise ValueError(f"Nifty benchmark merge failed for {missing_nifty_ratio:.1%} of stock rows.")
    df[benchmark_cols] = df[benchmark_cols].ffill().bfill()
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    def calculate_indicators(g):
        g['Vol_30D_SMA'] = g['Volume'].rolling(30, min_periods=10).mean()
        g['Del_30D_SMA'] = g['Delivery_Volume'].rolling(30, min_periods=10).mean()
        
        g['Vol_Surge'] = g['Volume'] / (g['Vol_30D_SMA'] + 1e-8)
        g['Del_Surge'] = g['Delivery_Volume'] / (g['Del_30D_SMA'] + 1e-8)
        
        if 'Turnover' in g.columns and not g['Turnover'].isna().all():
            g['VWAP'] = (g['Turnover'] * 100000) / (g['Volume'] + 1e-8)
        else:
            g['VWAP'] = (g['High'] + g['Low'] + g['Close']) / 3
            
        g['Daily_Range'] = g['High'] - g['Low']
        g['Close_Location'] = (g['Close'] - g['Low']) / (g['Daily_Range'] + 1e-8)
        g['Top_10_Percent'] = g['Close_Location'] >= 0.90
        
        g['Screener_Hit'] = (
            ((g['Vol_Surge'] > 2.0) | (g['Del_Surge'] > 2.0)) & 
            (g['Close'] > g['VWAP']) & 
            g['Top_10_Percent']
        )
        
        for window in [1, 5, 20]:
            stock_ret = g['Close'].pct_change(window)
            nifty_ret = g['Nifty_Close'].pct_change(window)
            g[f'RS_{window}D'] = stock_ret - nifty_ret
            
        try:
            g['ATR_30'] = ta.volatility.average_true_range(g['High'], g['Low'], g['Close'], window=30, fillna=True)
            indicator_bb = ta.volatility.BollingerBands(close=g['Close'], window=30, window_dev=2, fillna=True)
            g['BB_Width'] = indicator_bb.bollinger_wband()
        except:
            g['ATR_30'] = np.nan
            g['BB_Width'] = np.nan
            
        g['Nifty_vs_EMA50'] = g['Nifty_Close'] / (g['Nifty_EMA50'] + 1e-8) - 1
        g['Nifty_vs_EMA200'] = g['Nifty_Close'] / (g['Nifty_EMA200'] + 1e-8) - 1
        g['VIX_Change_5D'] = g['India_VIX'].pct_change(5)
        g['VIX_Rank_252D'] = g['India_VIX'].rolling(252, min_periods=20).rank(pct=True)
        
        g['High_52W'] = g['High'].rolling(252, min_periods=20).max()
        g['Dist_From_52W_High'] = g['Close'] / (g['High_52W'] + 1e-8) - 1
        
        # --- SMART MONEY ACCUMULATION SCORE ---
        # 1. Delivery Percentage Anomaly
        g['Del_Pct'] = g['Delivery_Volume'] / (g['Volume'] + 1e-8)
        g['Del_Pct_5D'] = g['Del_Pct'].rolling(5).mean()
        g['Del_Pct_50D'] = g['Del_Pct'].rolling(50, min_periods=10).mean()
        g['Del_Anomaly'] = g['Del_Pct_5D'] / (g['Del_Pct_50D'] + 1e-8)
        
        # 2. Volume-Price Divergence (Up Vol vs Down Vol)
        up_vol = np.where(g['Close'] > g['Close'].shift(1), g['Volume'], 0)
        down_vol = np.where(g['Close'] < g['Close'].shift(1), g['Volume'], 0)
        g['Up_Vol_15D'] = pd.Series(up_vol, index=g.index).rolling(15, min_periods=5).sum()
        g['Down_Vol_15D'] = pd.Series(down_vol, index=g.index).rolling(15, min_periods=5).sum()
        g['Vol_Trend'] = g['Up_Vol_15D'] / (g['Down_Vol_15D'] + 1e-8)
        
        # 3. Volatility Contraction (Bollinger Squeeze Rank)
        if 'BB_Width' in g.columns:
            g['BB_Squeeze_Rank'] = g['BB_Width'].rolling(120, min_periods=30).rank(pct=True)
            g['Squeeze_Score'] = 1.0 - g['BB_Squeeze_Rank']
        else:
            g['Squeeze_Score'] = 0.0
            
        # 4. 50-EMA Defense
        g['EMA_50'] = g['Close'].ewm(span=50, adjust=False).mean()
        g['Above_EMA50'] = (g['Close'] > g['EMA_50']).astype(int)
        
        # Calculate Composite Score (0-100)
        score_del = np.clip((g['Del_Anomaly'] - 1.0) / 1.5, 0, 1) * 30    # Max 30 pts
        score_vol = np.clip((g['Vol_Trend'] - 1.0) / 2.0, 0, 1) * 30      # Max 30 pts
        score_sqz = g['Squeeze_Score'].fillna(0) * 30                     # Max 30 pts
        score_ema = g['Above_EMA50'] * 10                                 # Max 10 pts
        
        g['Smart_Money_Score'] = (score_del + score_vol + score_sqz + score_ema).round(2)
        
        # The Watchlist Trigger
        g['Smart_Money_Hit'] = (g['Smart_Money_Score'] >= 80) & (g['Screener_Hit'] == False)
        
        # Buy at the Open of the NEXT day after the signal
        g['Entry_Price'] = g['Open'].shift(-1)
        
        for i in range(1, 31):
            fut_high = g['High'].shift(-1)[::-1].rolling(i, min_periods=1).max()[::-1]
            fut_low = g['Low'].shift(-1)[::-1].rolling(i, min_periods=1).min()[::-1]
            g[f'Max_Return_{i}D'] = fut_high / (g['Entry_Price'] + 1e-8) - 1
            g[f'Max_Drawdown_{i}D'] = fut_low / (g['Entry_Price'] + 1e-8) - 1
            
        g['Target_Success'] = np.where(
            g['Entry_Price'].notna() & g['Max_Return_10D'].notna(),
            (g['Max_Return_10D'] > 0.05).astype(int),
            np.nan
        )
        
        return g
        
    processed_df = df.groupby('Symbol', group_keys=True).apply(calculate_indicators).reset_index(level=0)
    processed_df = processed_df.reset_index(drop=True)
    processed_df.to_parquet(PROCESSED_FILE, index=False)
    return processed_df

def write_label_diagnostics(df):
    hits = df[df['Screener_Hit'] == True].copy()
    if hits.empty:
        diagnostics = {
            'total_screener_hits': 0,
            'usable_training_labels': 0,
            'label_completeness': 0.0,
            'positive_label_rate': 0.0,
            'suspicious_return_rate_gt_100pct': 0.0,
            'suspicious_drawdown_rate_lt_minus_50pct': 0.0,
        }
    else:
        max_date = hits['Date'].max()
        mature_hits = hits[hits['Date'] <= (max_date - pd.Timedelta(days=15))]
        usable = mature_hits.dropna(subset=['Entry_Price', 'Max_Return_10D', 'Max_Drawdown_10D', 'Target_Success'])
        diagnostics = {
            'total_screener_hits': int(len(hits)),
            'mature_screener_hits': int(len(mature_hits)),
            'usable_training_labels': int(len(usable)),
            'label_completeness': float(len(usable) / len(mature_hits)) if len(mature_hits) else 0.0,
            'positive_label_rate': float(usable['Target_Success'].mean()) if len(usable) else 0.0,
            'suspicious_return_rate_gt_100pct': float((usable['Max_Return_10D'] > 1.0).mean()) if len(usable) else 0.0,
            'suspicious_drawdown_rate_lt_minus_50pct': float((usable['Max_Drawdown_10D'] < -0.5).mean()) if len(usable) else 0.0,
            'label_definition': 'Buy next trading day open after Screener_Hit; positive if Max_Return_10D exceeds 5%. Incomplete future labels are excluded from training.',
        }
    with open(LABEL_DIAGNOSTICS_FILE, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logging.info(f"Label Diagnostics: {diagnostics}")

def calculate_classification_metrics(y_true, preds, probabilities):
    metrics = {
        'accuracy': accuracy_score(y_true, preds),
        'balanced_accuracy': balanced_accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
    }
    if probabilities is not None and len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, probabilities)
        metrics['average_precision'] = average_precision_score(y_true, probabilities)
    else:
        metrics['roc_auc'] = 0.0
        metrics['average_precision'] = 0.0
    return {key: float(value) for key, value in metrics.items()}

def model_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return None

def train_model(df):
    logging.info("Training and Evaluating ML Models with Walk-Forward Validation...")
    train_data = df[df['Screener_Hit'] == True].copy()
    
    features = MODEL_FEATURES
    
    train_data = train_data.replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features + ['Target_Success'])
    
    max_date = train_data['Date'].max()
    train_data = train_data[train_data['Date'] <= (max_date - pd.Timedelta(days=15))]
    train_data = train_data.sort_values('Date')
    
    if len(train_data) < 50:
        logging.warning("Not enough Screener Hits to train a reliable model. Training a fallback dummy model.")
        model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
        X_dummy = pd.DataFrame(np.random.rand(10, len(features)), columns=features)
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        metrics_data = {
            'walk_forward_metrics': {},
            'walk_forward_scores': {},
            'selection_metric': SELECTION_METRIC,
            'features': features,
            'top_models': ['FallbackDummy'],
            'warning': 'Not enough screener hits to train a reliable model.'
        }
        with open(os.path.join(DATA_DIR, "ml_metrics.json"), "w") as f:
            json.dump(metrics_data, f)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        return model

    # Define models to test
    # Use Pipelines for models that require scaling
    models = {
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'NeuralNetwork': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))
        ]),
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=42))
        ])
    }
    
    # Walk-forward validation by year
    train_data['Year'] = train_data['Date'].dt.year
    years = sorted(train_data['Year'].unique())
    
    model_fold_metrics = {name: [] for name in models.keys()}
    
    # Need at least 2 years for walk forward
    if len(years) >= 2:
        total_steps = (len(years) - 1) * len(models)
        current_step = 0
        for i in range(1, len(years)):
            test_year = years[i]
            train_mask = train_data['Year'] < test_year
            test_mask = train_data['Year'] == test_year
            
            X_tr, y_tr = train_data.loc[train_mask, features], train_data.loc[train_mask, 'Target_Success']
            X_te, y_te = train_data.loc[test_mask, features], train_data.loc[test_mask, 'Target_Success']
            
            # Skip if very few test or train samples
            if len(y_tr) < 20 or len(y_te) < 10:
                continue
                
            for name, model in models.items():
                current_step += 1
                logging.info(f"PROGRESS:{current_step}/{total_steps}")
                logging.info(f"Evaluating {name} on {test_year} Out-Of-Sample Data...")
                try:
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_te)
                    probabilities = model_probabilities(model, X_te)
                    model_fold_metrics[name].append(calculate_classification_metrics(y_te, preds, probabilities))
                except Exception as e:
                    logging.warning(f"Model {name} failed on year {test_year}: {e}")
                    model_fold_metrics[name].append({
                        'accuracy': 0.0,
                        'balanced_accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'roc_auc': 0.0,
                        'average_precision': 0.0,
                    })
                    
        # Average each out-of-sample metric across walk-forward folds.
        avg_metrics = {}
        for name, folds in model_fold_metrics.items():
            if len(folds) > 0:
                avg_metrics[name] = {
                    metric: float(np.mean([fold[metric] for fold in folds]))
                    for metric in folds[0].keys()
                }
            else:
                avg_metrics[name] = {
                    'accuracy': 0.0,
                    'balanced_accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'roc_auc': 0.0,
                    'average_precision': 0.0,
                }
                
        logging.info(f"Walk-Forward Validation Metrics: {avg_metrics}")
        
        # Select models by average precision because the dashboard ranks trades by probability.
        sorted_models = sorted(avg_metrics.items(), key=lambda item: item[1][SELECTION_METRIC], reverse=True)
        top1_name = sorted_models[0][0]
        top2_name = sorted_models[1][0]
        logging.info(f"Selected Top 2 Models by {SELECTION_METRIC}: {top1_name} and {top2_name}")
        
        top1_model = models[top1_name]
        top2_model = models[top2_name]
        
        metrics_data = {
            'walk_forward_metrics': avg_metrics,
            'walk_forward_scores': {name: values[SELECTION_METRIC] for name, values in avg_metrics.items()},
            'selection_metric': SELECTION_METRIC,
            'features': features,
            'top_models': [top1_name, top2_name]
        }
        with open(os.path.join(DATA_DIR, "ml_metrics.json"), "w") as f:
            json.dump(metrics_data, f)
            
    else:
        # Not enough years for walk-forward, default to XGBoost and RF
        logging.warning("Not enough years for walk-forward CV. Defaulting to XGBoost and RandomForest.")
        top1_model = models['XGBoost']
        top2_model = models['RandomForest']
        top1_name, top2_name = 'XGBoost', 'RandomForest'

        metrics_data = {
            'walk_forward_metrics': {
                name: {
                    'accuracy': 0.0,
                    'balanced_accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'roc_auc': 0.0,
                    'average_precision': 0.0,
                }
                for name in models.keys()
            },
            'walk_forward_scores': {'XGBoost': 0, 'RandomForest': 0, 'NeuralNetwork': 0, 'LogisticRegression': 0, 'SVM': 0},
            'selection_metric': SELECTION_METRIC,
            'features': features,
            'top_models': [top1_name, top2_name]
        }
        with open(os.path.join(DATA_DIR, "ml_metrics.json"), "w") as f:
            json.dump(metrics_data, f)

    # Train final ensemble model on ALL data using the top 2 algorithms
    logging.info(f"Training final Ensemble ({top1_name} & {top2_name}) on full dataset...")
    X_full = train_data[features]
    y_full = train_data['Target_Success']
    
    ensemble = VotingClassifier(
        estimators=[(top1_name, top1_model), (top2_name, top2_model)],
        voting='soft'
    )
    
    ensemble.fit(X_full, y_full)
    
    # Evaluate on whole set only as an overfit diagnostic; model selection uses walk-forward metrics above.
    preds_full = ensemble.predict(X_full)
    prob_full = model_probabilities(ensemble, X_full)
    train_metrics = calculate_classification_metrics(y_full, preds_full, prob_full)
    logging.info(f"Final Ensemble Training Diagnostics: {train_metrics}")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(ensemble, f)
        
    return ensemble

def main():
    logging.info("Starting Institutional Volume Screener Pipeline")
    
    DAYS_TO_FETCH = int(os.environ.get('DAYS_TO_FETCH', 1250))
    logging.info(f"Configured to fetch {DAYS_TO_FETCH} trading days.")
    
    nifty_df = fetch_nifty_benchmark(days_back=DAYS_TO_FETCH)
    corporate_actions = fetch_corporate_actions(days_back=DAYS_TO_FETCH)
    raw_df = fetch_historical_bhavcopies(days_back=DAYS_TO_FETCH, nifty_df=nifty_df)
    
    if raw_df.empty:
        logging.error("Failed to fetch equity data.")
        return
        
    adjusted_df = apply_corporate_action_adjustments(raw_df, corporate_actions)
    processed_df = engineer_features(adjusted_df, nifty_df)
    write_label_diagnostics(processed_df)
    model = train_model(processed_df)
    
    logging.info("Pipeline Execution Completed. Run 'streamlit run app.py' to view the dashboard.")

if __name__ == "__main__":
    main()
