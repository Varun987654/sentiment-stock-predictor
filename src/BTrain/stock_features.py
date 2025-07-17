# src/stock_features.py

import os
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────
INPUT_DIR  = "data/03_StockPrices"
OUTPUT_DIR = INPUT_DIR  # overwrite in place

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fn in os.listdir(INPUT_DIR):
    if not fn.endswith(".csv"):
        continue

    infile  = os.path.join(INPUT_DIR, fn)
    outfile = os.path.join(OUTPUT_DIR, fn)

    # 1) Load & lowercase column names
    df = pd.read_csv(infile, parse_dates=["bin_end_time"])
    df.columns = df.columns.str.strip().str.lower()  # normalize

    # 2) Ensure we now have the key columns
    required = {"open","high","low","close","volume"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(f"{fn} is missing columns: {missing}")

    # 3) Sort by time
    df = df.sort_values("bin_end_time").reset_index(drop=True)

    # 4) Compute returns and changes
    df["bin_return"]     = df["close"].pct_change().fillna(0.0)
    df["volume_change"]  = df["volume"].pct_change().fillna(0.0)
    df["bin_range"]      = df["high"] - df["low"]

    # 5) Save back
    #   We reattach the original capitalized columns if you need them, but
    #   most downstream code should refer to lowercase names now.
    df.to_csv(outfile, index=False)
    print(f"✔ Updated {fn}: added bin_return, volume_change, bin_range")
