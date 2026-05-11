import pandas as pd
import numpy as np
import os

def run_deeper_forensics():
    if not os.path.exists('data/oos_predictions.parquet'):
        return
        
    oos = pd.read_parquet('data/oos_predictions.parquet')
    oos['Date'] = pd.to_datetime(oos['Date'])
    oos['Year'] = oos['Date'].dt.year
    
    # All-Weather Rolling Rank (Last 365 days window for relative top 5%)
    oos['Yearly_Rank'] = oos.groupby('Year')['OOS_Score'].rank(pct=True)
    signals = oos[oos['Yearly_Rank'] >= 0.95].copy()
    
    print("\n" + "="*60)
    print("🎯 MULTIBAGGER ENGINE: PER-TRADE ALPHA FORENSICS")
    print("="*60)
    
    stats = []
    for year in range(2021, 2026):
        y_data = signals[signals['Year'] == year]
        if y_data.empty: continue
        
        trades = len(y_data)
        win_rate = y_data['Target_Success'].mean()
        avg_ret = y_data['Realized_Return'].mean()
        
        # Wins vs Losses
        wins = y_data[y_data['Realized_Return'] > 0]['Realized_Return']
        losses = y_data[y_data['Realized_Return'] <= 0]['Realized_Return']
        
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = losses.mean() if not losses.empty else 0
        
        # Risk/Reward Ratio (RR)
        rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        stats.append({
            'Year': year,
            'Trades': trades,
            'Win Rate': f"{win_rate:.1%}",
            'Avg Win': f"+{avg_win:.2%}",
            'Avg Loss': f"{avg_loss:.2%}",
            'RR Ratio': f"{rr_ratio:.2f}x",
            'Expectancy': f"{avg_ret:+.2%}"
        })
        
    df_stats = pd.DataFrame(stats)
    print(df_stats.to_string(index=False))
    print("="*60)
    
    # Global Metrics
    global_avg_win = signals[signals['Realized_Return'] > 0]['Realized_Return'].mean()
    global_avg_loss = signals[signals['Realized_Return'] <= 0]['Realized_Return'].mean()
    global_win_rate = (signals['Realized_Return'] > 0).mean()
    
    print(f"OVERALL WIN RATE: {global_win_rate:.1%}")
    print(f"OVERALL AVG WIN: +{global_avg_win:.2%}")
    print(f"OVERALL AVG LOSS: {global_avg_loss:.2%}")
    print(f"OVERALL RR RATIO: {abs(global_avg_win/global_avg_loss):.2f}x")
    print(f"GLOBAL EXPECTANCY PER TRADE: {signals['Realized_Return'].mean():+.2%}")
    print("="*60)

if __name__ == "__main__":
    run_deeper_forensics()
