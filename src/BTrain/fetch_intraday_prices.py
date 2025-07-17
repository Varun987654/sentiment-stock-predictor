import os
import yfinance as yf
import pandas as pd

# 1. Configuration
TICKERS   = ["AAPL", "GOOGL", "TSLA"]
START     = "2025-04-23"
END       = "2025-06-17"
INTERVAL  = "30m"
OUT_DIR   = "data/03_StockPrices"

# 2. Ensure output folder exists
os.makedirs(OUT_DIR, exist_ok=True)

# 3. Loop and fetch
for sym in TICKERS:
    print(f"Fetching {sym} {INTERVAL} bars from {START} to {END}...")
    ticker = yf.Ticker(sym)
    df = ticker.history(
        interval=INTERVAL,
        start=START,
        end=END
    )
    # Rename index to column for easy merge later
    df = df.reset_index().rename(columns={"Datetime": "bin_end_time"})
    # Keep only the needed columns
    df = df[["bin_end_time", "Open", "High", "Low", "Close", "Volume"]]
    # Save to CSV
    out_path = f"{OUT_DIR}/price_{sym}.csv"
    df.to_csv(out_path, index=False)
    print(f" → Saved {len(df)} rows to {out_path}")
