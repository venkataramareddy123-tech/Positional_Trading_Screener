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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# Suppress pandas fragmentation warnings during the 30-day loop
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
RAW_FILE = os.path.join(DATA_DIR, "raw_data.parquet")
NIFTY_FILE = os.path.join(DATA_DIR, "nifty_data.parquet")
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_data.parquet")
MODEL_FILE = os.path.join(DATA_DIR, "xgb_model.pkl")

def get_trading_days(start_date, end_date):
    days = pd.date_range(start=start_date, end=end_date, freq='B')
    return [d.strftime('%d-%m-%Y') for d in days]

def fetch_historical_bhavcopies(days_back=1250):
    """Fetches Bhavcopy with delivery for the last N trading days."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back * 1.5)
    
    trading_dates = get_trading_days(start_date, end_date)[-days_back:]
    all_data = []
    
    if os.path.exists(RAW_FILE):
        logging.info("Loading existing raw data...")
        df_existing = pd.read_parquet(RAW_FILE)
        if 'Date' in df_existing.columns:
            existing_dates = df_existing['Date'].dt.strftime('%d-%m-%Y').unique().tolist()
        else:
            existing_dates = []
    else:
        df_existing = pd.DataFrame()
        existing_dates = []

    dates_to_fetch = [d for d in trading_dates if d not in existing_dates]
    total_dates = len(dates_to_fetch)
    new_data_fetched = False
    
    for i, date_str in enumerate(dates_to_fetch):
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
                df_day[col] = pd.to_numeric(df_day[col].astype(str).str.replace(',', '').str.replace('-', ''), errors='coerce')
                
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
    
    nifty = yf.download('^NSEI', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    nifty = nifty.reset_index()
    
    if nifty.empty:
        logging.warning("Failed to fetch Nifty data.")
        return pd.DataFrame()
        
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [col[0] if col[0] != 'Date' else 'Date' for col in nifty.columns]
        
    nifty = nifty.rename(columns={'Close': 'Nifty_Close'})
    nifty['Date'] = pd.to_datetime(nifty['Date']).dt.tz_localize(None)
    
    nifty['Nifty_EMA50'] = nifty['Nifty_Close'].ewm(span=50, adjust=False).mean()
    nifty['Nifty_EMA200'] = nifty['Nifty_Close'].ewm(span=200, adjust=False).mean()
    
    vix = yf.download('^INDIAVIX', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    vix = vix.reset_index()
    if not vix.empty:
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [col[0] if col[0] != 'Date' else 'Date' for col in vix.columns]
        vix = vix.rename(columns={'Close': 'India_VIX'})
        vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
        nifty = pd.merge(nifty, vix[['Date', 'India_VIX']], on='Date', how='left')
    else:
        nifty['India_VIX'] = np.nan
        
    nifty = nifty[['Date', 'Nifty_Close', 'Nifty_EMA50', 'Nifty_EMA200', 'India_VIX']]
    nifty.to_parquet(NIFTY_FILE, index=False)
    return nifty

def adjust_group_vectorized(df):
    df = df.sort_values('Date').reset_index(drop=True)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Ratio'] = df['Prev_Close'] / (df['Open'] + 1e-8)
    
    common_ratios = np.array([2.0, 3.0, 4.0, 5.0, 10.0, 1.5, 1.25, 1.333, 2.5])
    df['Adj_Factor'] = 1.0
    mask = df['Ratio'] > 1.15
    
    if mask.any():
        ratios = df.loc[mask, 'Ratio'].values
        diff = np.abs(ratios[:, None] - common_ratios)
        min_idx = np.argmin(diff, axis=1)
        closest_ratios = common_ratios[min_idx]
        
        valid = (np.abs(closest_ratios - ratios) / closest_ratios) < 0.1
        df.loc[df.index[mask][valid], 'Adj_Factor'] = closest_ratios[valid]
        
    df['Cum_Adj'] = df['Adj_Factor'].iloc[::-1].cumprod().iloc[::-1]
    df['Cum_Adj'] = df['Cum_Adj'].shift(-1).fillna(1.0)
    
    df['Open'] = df['Open'] / df['Cum_Adj']
    df['High'] = df['High'] / df['Cum_Adj']
    df['Low'] = df['Low'] / df['Cum_Adj']
    df['Close'] = df['Close'] / df['Cum_Adj']
    df['Volume'] = df['Volume'] * df['Cum_Adj']
    df['Delivery_Volume'] = df['Delivery_Volume'] * df['Cum_Adj']
    
    return df.drop(columns=['Prev_Close', 'Ratio', 'Adj_Factor', 'Cum_Adj'])

def apply_corporate_action_adjustments(df):
    logging.info("Applying Corporate Action Adjustments (Splits/Bonuses)...")
    adjusted_df = df.groupby('Symbol', group_keys=True).apply(adjust_group_vectorized).reset_index(level=0)
    return adjusted_df.reset_index(drop=True)

def engineer_features(df, nifty_df):
    logging.info("Engineering Features and Screener Logic...")
    
    df['Date'] = pd.to_datetime(df['Date'])
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
    
    df = pd.merge(df, nifty_df, on='Date', how='left')
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
            
        g['Target_Success'] = (g['Max_Return_10D'] > 0.05).astype(int)
        
        return g
        
    processed_df = df.groupby('Symbol', group_keys=True).apply(calculate_indicators).reset_index(level=0)
    processed_df = processed_df.reset_index(drop=True)
    processed_df.to_parquet(PROCESSED_FILE, index=False)
    return processed_df

def train_model(df):
    logging.info("Training and Evaluating ML Models with Walk-Forward Validation...")
    train_data = df[df['Screener_Hit'] == True].copy()
    
    features = [
        'Vol_Surge', 'Del_Surge', 'Close_Location',
        'RS_1D', 'RS_5D', 'RS_20D', 'ATR_30', 'BB_Width',
        'Nifty_vs_EMA50', 'Nifty_vs_EMA200', 'Dist_From_52W_High',
        'Smart_Money_Score'
    ]
    
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
    
    model_scores = {name: [] for name in models.keys()}
    
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
                    acc = accuracy_score(y_te, preds)
                    model_scores[name].append(acc)
                except Exception as e:
                    logging.warning(f"Model {name} failed on year {test_year}: {e}")
                    model_scores[name].append(0)
                    
        # Average the scores
        avg_scores = {}
        for name, scores in model_scores.items():
            if len(scores) > 0:
                avg_scores[name] = np.mean(scores)
            else:
                avg_scores[name] = 0
                
        logging.info(f"Walk-Forward Validation Accuracy Scores: {avg_scores}")
        
        # Select Top 2 models
        sorted_models = sorted(avg_scores.items(), key=lambda item: item[1], reverse=True)
        top1_name = sorted_models[0][0]
        top2_name = sorted_models[1][0]
        logging.info(f"Selected Top 2 Models: {top1_name} and {top2_name}")
        
        top1_model = models[top1_name]
        top2_model = models[top2_name]
        
        metrics_data = {
            'walk_forward_scores': avg_scores,
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
            'walk_forward_scores': {'XGBoost': 0, 'RandomForest': 0, 'NeuralNetwork': 0, 'LogisticRegression': 0, 'SVM': 0},
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
    
    # Evaluate on whole set just for reporting
    preds_full = ensemble.predict(X_full)
    logging.info(f"Final Ensemble Model Training Accuracy: {accuracy_score(y_full, preds_full):.2f}")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(ensemble, f)
        
    return ensemble

def main():
    logging.info("Starting Institutional Volume Screener Pipeline")
    
    DAYS_TO_FETCH = int(os.environ.get('DAYS_TO_FETCH', 1250))
    logging.info(f"Configured to fetch {DAYS_TO_FETCH} trading days.")
    
    nifty_df = fetch_nifty_benchmark(days_back=DAYS_TO_FETCH)
    raw_df = fetch_historical_bhavcopies(days_back=DAYS_TO_FETCH)
    
    if raw_df.empty:
        logging.error("Failed to fetch equity data.")
        return
        
    adjusted_df = apply_corporate_action_adjustments(raw_df)
    processed_df = engineer_features(adjusted_df, nifty_df)
    model = train_model(processed_df)
    
    logging.info("Pipeline Execution Completed. Run 'streamlit run app.py' to view the dashboard.")

if __name__ == "__main__":
    main()
