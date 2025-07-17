import os
import yfinance as yf
import pandas as pd

# 1. Configuration
START_DATE = "2025-04-23"
END_DATE   = "2025-06-17"
OUT_DIR    = "data/03_Macros"

# Tickers: Gold futures and Brent crude
MACRO_TICKERS = {
    "gold_price": "GC=F",
    "brent_oil":  "BZ=F"
}

# 2. Ensure output folder exists
os.makedirs(OUT_DIR, exist_ok=True)

# 3. Fetch daily data and save both in one go
all_macro = {}
for name, ticker_symbol in MACRO_TICKERS.items():
    print(f"Downloading {name} ({ticker_symbol}) from {START_DATE} to {END_DATE}…")
    df = yf.Ticker(ticker_symbol).history(
        interval="1d",
        start=START_DATE,
        end=END_DATE
    )[['Close']].rename(columns={"Close": name})
    df.index = pd.to_datetime(df.index).date  # Convert to plain date
    all_macro[name] = df

# 4. Merge into a single DataFrame on date
macro_df = pd.concat(all_macro.values(), axis=1)

# 5. Forward-fill any missing days
macro_df = macro_df.ffill().bfill()

# 6. Save to CSV
out_path = f"{OUT_DIR}/macro_gold_brent.csv"
macro_df.to_csv(out_path, index_label="date")
print(f"Saved macro indicators to {out_path}")
