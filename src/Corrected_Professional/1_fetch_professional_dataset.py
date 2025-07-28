# src/Corrected_Professional/1_fetch_professional_dataset.py

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- CONFIG ---
TICKERS   = ["AAPL", "GOOGL", "TSLA"]
# Fetch data for the last 2 years
END_DATE  = datetime.now()
START_DATE = END_DATE - timedelta(days=729)
INTERVAL  = "30m"
# Saving to a new, separate data folder
OUT_DIR   = "data/03_StockPrices_Professional" 

print(f"--- Starting Professional Data Fetch (2 Years) ---")
os.makedirs(OUT_DIR, exist_ok=True)

start_str = START_DATE.strftime('%Y-%m-%d')
end_str = END_DATE.strftime('%Y-%m-%d')

for sym in TICKERS:
    print(f"Fetching {sym} {INTERVAL} bars from {start_str} to {end_str}...")
    
    df = yf.download(
        tickers=sym,
        start=start_str,
        end=end_str,
        interval=INTERVAL
    )
    
    if df.empty:
        print(f"   -> No data found for {sym}. Skipping.")
        continue
        
    df = df.reset_index().rename(columns={"Datetime": "bin_end_time"})
    df = df[["bin_end_time", "Open", "High", "Low", "Close", "Volume"]]
    
    out_path = f"{OUT_DIR}/price_{sym}.csv"
    df.to_csv(out_path, index=False)
    print(f" → Saved {len(df)} rows to {out_path}")

print("\n--- Professional Data Fetch Complete ---")