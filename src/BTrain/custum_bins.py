# src/custom_bins.py

import os
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────
INPUT_CSV  = "data/01_RawNewsHeadlines/FINAL.csv"
OUT_CSV    = "data/05_Binned_sentiments/ticker_session_bins.csv"
START_DATE = "2025-04-23"
END_DATE   = "2025-06-17"

# ─── PREP ──────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# 1) Load & drop invalid rows
df = pd.read_csv(INPUT_CSV, parse_dates=["publishedAt"])
df = df.dropna(subset=["ticker", "publishedAt"])

# 2) Ensure every timestamp ends up in UTC without double-localizing
def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    # if naive, assume it's UTC; if already tz-aware, convert to UTC
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

df["publishedAt"] = df["publishedAt"].apply(to_utc)

# 3) Define 3 sessions in UTC
def session_for(ts: pd.Timestamp) -> str:
    # minutes since midnight
    m = ts.hour*60 + ts.minute
    if 21*60 <= m < 24*60:
        return "after_close"
    if 0 <= m < 13*60 + 30:
        return "pre_open"
    if 13*60 + 30 <= m < 20*60:
        return "trading"
    return None

# 4) Attach date/session and filter business days
biz_days = pd.date_range(START_DATE, END_DATE, freq="B", tz=None).date
df["date"]    = df["publishedAt"].dt.date
df["session"] = df["publishedAt"].apply(session_for)
df = df[df["date"].isin(biz_days)]

# 5) Aggregate sentiment per ticker–date–session
agg = (
    df
    .groupby(["ticker","date","session"])["compound"]
    .agg(
        compound_count="count",
        compound_mean="mean",
        compound_std="std",
        compound_max="max",
        compound_min="min"
    )
    .reset_index()
)

# 6) Reindex so each (ticker, date) has all three sessions
idx = pd.MultiIndex.from_product(
    [df["ticker"].unique(), biz_days, ["after_close","pre_open","trading"]],
    names=["ticker","date","session"]
)
agg = (
    agg
    .set_index(["ticker","date","session"])
    .reindex(idx, fill_value=0)
    .reset_index()
)

# 7) Save to CSV
agg.to_csv(OUT_CSV, index=False)
print(f"Saved session-binned sentiment → {OUT_CSV}")
