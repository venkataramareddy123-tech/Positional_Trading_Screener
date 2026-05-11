import pandas as pd
import numpy as np
import os

def run_2022_diversified_audit():
    if not os.path.exists('data/oos_predictions.parquet') or not os.path.exists('data/nifty_data.parquet'):
        print("Data missing.")
        return
        
    oos = pd.read_parquet('data/oos_predictions.parquet')
    oos['Date'] = pd.to_datetime(oos['Date'])
    oos['Exit_Date'] = pd.to_datetime(oos['Exit_Date'])
    
    # Start strictly from 2022-01-01
    oos = oos[oos['Date'] >= '2022-01-01'].sort_values('Date')
    
    nifty = pd.read_parquet('data/nifty_data.parquet')
    nifty['Date'] = pd.to_datetime(nifty['Date'])
    nifty = nifty[nifty['Date'] >= '2022-01-01'].sort_values('Date')
    
    # Dynamic Rank: Top 5% of signals available at that time
    oos['Year'] = oos['Date'].dt.year
    oos['Yearly_Rank'] = oos.groupby('Year')['OOS_Score'].rank(pct=True)
    signals = oos[oos['Yearly_Rank'] >= 0.95].copy()
    signals = signals.sort_values(['Date', 'OOS_Score'], ascending=[True, False])
    
    # Portfolio Settings
    INITIAL_CAPITAL = 100000.0
    MAX_POSITIONS = 20 
    
    equity = INITIAL_CAPITAL
    active_positions = []
    daily_equity = []
    
    all_dates = sorted(pd.concat([pd.Series(signals['Date'].unique()), pd.Series(nifty['Date'].unique())]).unique())

    for current_date in all_dates:
        # 1. Check for Exits
        remaining_positions = []
        for pos in active_positions:
            if current_date >= pos['exit_date']:
                # Capital rotation: Add the profit/loss back to the pool
                trade_profit = pos['trade_capital'] * pos['realized_return']
                equity += trade_profit
            else:
                remaining_positions.append(pos)
        active_positions = remaining_positions
        
        # 2. Check for Entries
        available_slots = MAX_POSITIONS - len(active_positions)
        if available_slots > 0:
            todays_signals = signals[signals['Date'] == current_date]
            for _, sig in todays_signals.head(available_slots).iterrows():
                # For fixed 5K per trade as requested:
                trade_capital = 5000.0 
                active_positions.append({
                    'exit_date': sig['Exit_Date'],
                    'realized_return': sig['Realized_Return'],
                    'trade_capital': trade_capital 
                })
        
        daily_equity.append({'Date': current_date, 'Equity': equity})

    # Analysis
    audit_df = pd.DataFrame(daily_equity)
    audit_df['Year'] = audit_df['Date'].dt.year
    
    print("\n" + "="*85)
    print("📈 DIVERSIFIED MULTIBAGGER AUDIT: 2022 BEAR MARKET START")
    print(f"Initial Capital: ₹100,000 | Allocation: ₹5,000 per trade ({MAX_POSITIONS} Slots)")
    print("="*85)
    
    results = []
    for year in range(2022, 2027):
        y_data = audit_df[audit_df['Year'] == year]
        if y_data.empty: continue
        
        start_val = y_data.iloc[0]['Equity']
        end_val = y_data.iloc[-1]['Equity']
        
        y_nifty = nifty[nifty['Date'].dt.year == year]
        nifty_ret = (y_nifty.iloc[-1]['Nifty_Close'] / y_nifty.iloc[0]['Nifty_Close']) - 1 if not y_nifty.empty else 0
        
        results.append({
            'Year': year,
            'Portfolio_Start': f"₹{start_val:,.0f}",
            'Portfolio_End': f"₹{end_val:,.0f}",
            'Strategy_Return': f"{(end_val/start_val - 1):+.2%}",
            'Nifty_Return': f"{nifty_ret:+.2%}",
            'Alpha': f"{(end_val/start_val - 1 - nifty_ret):+.2%}"
        })
        
    print(pd.DataFrame(results).to_string(index=False))
    print("="*85)
    total_strat_ret = (equity/INITIAL_CAPITAL-1)
    nifty_start = nifty.iloc[0]['Nifty_Close']
    nifty_end = nifty.iloc[-1]['Nifty_Close']
    total_nifty_ret = (nifty_end/nifty_start - 1)
    
    print(f"FINAL PORTFOLIO VALUE: ₹{equity:,.0f} ({total_strat_ret*100:+.2f}%)")
    print(f"TOTAL NIFTY RETURN (2022-Current): {total_nifty_ret*100:+.2f}%")
    print(f"NET ALPHA GENERATED: {(total_strat_ret - total_nifty_ret)*100:+.2f}%")
    print("="*85)

if __name__ == "__main__":
    run_2022_diversified_audit()
