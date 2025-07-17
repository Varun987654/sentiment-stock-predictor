import os
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────
PRICE_DIR     = "data/03_StockPrices"
SENTIMENT_CSV = "data/05_Binned_sentiments/ticker_session_bins.csv"
MACRO_CSV     = "data/03_Macros/macro_gold_brent.csv"
OUTPUT_CSV    = "data/06_Features/merged.csv"

# ensure output folder exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# 1) Load and normalize session-binned sentiment
sess = pd.read_csv(SENTIMENT_CSV, parse_dates=["date"])
sess["date"] = sess["date"].dt.normalize()

# 2) Pivot to wide form: one row per (ticker, date) with 3×5 sentiment stats
wide = (
    sess.pivot_table(
        index=["ticker","date"],
        columns="session",
        values=["compound_count","compound_mean","compound_std","compound_max","compound_min"],
        fill_value=0
    )
)
# Flatten MultiIndex columns: ('compound_mean','trading') -> 'trading_mean'
wide.columns = [f"{session}_{stat}" for stat, session in wide.columns]
wide = wide.reset_index()

# 3) Load daily macro and normalize date
macro = pd.read_csv(MACRO_CSV, parse_dates=["date"])
macro["date"] = macro["date"].dt.normalize()

# 4) Timezone & session helpers
def to_et_minute(dt_series):
    et = dt_series.dt.tz_convert("US/Eastern")
    return et.dt.hour * 60 + et.dt.minute

def session_for(minute):
    if 4*60 <= minute < 9*60 + 30:
        return "pre_open"
    elif 9*60 + 30 <= minute < 16*60:
        return "trading"
    elif 16*60 <= minute < 20*60:
        return "after_close"
    return None

merged_list = []

# 5) Loop through each ticker's price file
for fname in sorted(os.listdir(PRICE_DIR)):
    if not fname.endswith(".csv"):
        continue

    ticker = fname.split("_")[1].split(".")[0]
    price  = pd.read_csv(
        os.path.join(PRICE_DIR, fname),
        parse_dates=["bin_end_time"]
    )

    # Ensure UTC tz-awareness
    ts = pd.to_datetime(price["bin_end_time"])
    if ts.dt.tz is None:
        price["bin_end_time"] = ts.dt.tz_localize("UTC")
    else:
        price["bin_end_time"] = ts.dt.tz_convert("UTC")

    # Compute date & session
    et_minutes = to_et_minute(price["bin_end_time"])
    et_dt      = price["bin_end_time"].dt.tz_convert("US/Eastern")
    price["date"]    = et_dt.dt.normalize().dt.tz_localize(None)
    price["session"] = et_minutes.map(session_for)
    price["ticker"]  = ticker

    # 6) Merge wide-session sentiment
    df = price.merge(
        wide,
        on=["ticker","date"],
        how="left"
    )
    # Fill any missing sentiment values
    for col in wide.columns:
        if col not in ("ticker","date"):
            df[col] = df[col].fillna(0.0)

    # 7) Merge macro and forward/backfill
    df = (
        df.merge(macro, on="date", how="left")
          .sort_values("bin_end_time")
          .ffill()
          .bfill()
    )

    merged_list.append(df)

# 8) Concatenate and save
final = pd.concat(merged_list, ignore_index=True)
final.to_csv(OUTPUT_CSV, index=False)
print(f"✔ Wrote merged features with pre_open/trading/after_close → {OUTPUT_CSV}")
