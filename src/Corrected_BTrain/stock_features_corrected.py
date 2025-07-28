# src/Corrected_BTrain/stock_features_corrected.py

import os
import pandas as pd

# --- CONFIG ---
INPUT_DIR  = "data/03_StockPrices"
OUTPUT_DIR = "data/04_Initial_Features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fn in os.listdir(INPUT_DIR):
    if not fn.endswith(".csv"):
        continue

    infile  = os.path.join(INPUT_DIR, fn)
    outfile = os.path.join(OUTPUT_DIR, fn)
    
    print(f"Processing {fn}...")

    # 1) Load original price data
    df = pd.read_csv(infile, parse_dates=["bin_end_time"])

    # --- FIX: Add the ticker symbol as a column from the filename ---
    ticker_symbol = fn.replace('price_', '').replace('.csv', '')
    df['ticker'] = ticker_symbol
    
    df.columns = df.columns.str.strip().str.lower()

    # 2) Sort by time
    df = df.sort_values("bin_end_time").reset_index(drop=True)

    # 3) Compute LAGGED features
    df["bin_return"] = (df["close"] / df["open"]) - 1
    df["lagged_return"]     = df["bin_return"].shift(1)
    df["lagged_volume_change"]  = df["volume"].pct_change().shift(1)
    df["lagged_bin_range"]      = (df["high"] - df["low"]).shift(1)

    # 4) Fill missing values
    df.fillna(0.0, inplace=True)

    # 5) Save the new file to the new directory
    df.to_csv(outfile, index=False)
    print(f"✔ Saved corrected features for {fn} to {outfile}")