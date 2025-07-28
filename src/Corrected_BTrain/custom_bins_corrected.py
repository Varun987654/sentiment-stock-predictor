# src/Corrected_BTrain/custom_bins_corrected.py

import os
import pandas as pd
import numpy as np

# --- CONFIG ---
NEWS_CSV = "data/01_RawNewsHeadlines/FINAL.csv"
PRICE_DIR = "data/03_StockPrices"
OUT_CSV = "data/05_Binned_sentiments_corrected/sentiment_features_corrected.csv"
START_DATE = "2025-04-23"
END_DATE = "2025-06-17"
TICKERS = ["AAPL", "GOOGL", "TSLA"]

# --- PREP ---
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# 1) Load and process raw news headlines
news_df = pd.read_csv(NEWS_CSV, parse_dates=["publishedAt"])
news_df = news_df.dropna(subset=["ticker", "publishedAt"])

def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None: return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
news_df["publishedAt"] = news_df["publishedAt"].apply(to_utc)

# --- PART 1: DAILY PRE-OPEN & LAGGED AFTER-CLOSE SENTIMENT ---

def session_for(ts: pd.Timestamp) -> str:
    m = ts.hour * 60 + ts.minute
    if 21*60 <= m < 24*60: return "after_close"
    if 0 <= m < 13*60 + 30: return "pre_open"
    if 13*60 + 30 <= m < 20*60: return "trading"
    return None

biz_days = pd.to_datetime(pd.date_range(START_DATE, END_DATE, freq="B")).date
news_df["date"] = news_df["publishedAt"].dt.date
news_df["session"] = news_df["publishedAt"].apply(session_for)
daily_df = news_df[news_df["date"].isin(biz_days)]

daily_agg = daily_df[daily_df['session'].isin(['pre_open', 'after_close'])].groupby(["ticker", "date", "session"])["compound"].agg(
    count="count", mean="mean"
).unstack(level='session').fillna(0)
daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]
after_close_cols = [c for c in daily_agg.columns if 'after_close' in c]
daily_agg[after_close_cols] = daily_agg.groupby('ticker')[after_close_cols].shift(1).fillna(0)
daily_agg = daily_agg.reset_index()

# --- PART 2: CUMULATIVE INTRA-DAY TRADING SENTIMENT ---

all_bins = []
for ticker in TICKERS:
    price_df = pd.read_csv(f"{PRICE_DIR}/price_{ticker}.csv", parse_dates=['bin_end_time'])
    price_df['ticker'] = ticker
    all_bins.append(price_df[['bin_end_time', 'ticker']])
time_grid_df = pd.concat(all_bins).sort_values('bin_end_time').reset_index(drop=True)
time_grid_df['bin_end_time'] = time_grid_df['bin_end_time'].apply(to_utc)
time_grid_df['date'] = time_grid_df['bin_end_time'].dt.date

# Create a generic 'timestamp' column for sorting
time_grid_df_merged = time_grid_df.copy()
time_grid_df_merged.rename(columns={'bin_end_time': 'timestamp'}, inplace=True)

trading_news_df = news_df[news_df['session'] == 'trading'][['publishedAt', 'ticker', 'compound', 'date']]
trading_news_df.rename(columns={'publishedAt': 'timestamp'}, inplace=True)

combined_df = pd.concat([time_grid_df_merged, trading_news_df]).sort_values(by=['ticker', 'timestamp']).reset_index(drop=True)

expanding_aggs_list = []
for group_keys, group_df in combined_df.groupby(['ticker', 'date']):
    group_df['trading_compound_count'] = group_df['compound'].expanding().count()
    group_df['trading_compound_mean'] = group_df['compound'].expanding().mean()
    expanding_aggs_list.append(group_df)
    
cumulative_df = pd.concat(expanding_aggs_list)
cumulative_df[['trading_compound_count', 'trading_compound_mean']] = cumulative_df[['trading_compound_count', 'trading_compound_mean']].ffill()
cumulative_df = cumulative_df.fillna(0)

# **FIX**: Use left_on and right_on to specify the correct column names for the merge
final_intra_day_sentiment = pd.merge(
    time_grid_df,
    cumulative_df,
    left_on=['ticker', 'date', 'bin_end_time'],
    right_on=['ticker', 'date', 'timestamp'],
    how='left'
)
final_intra_day_sentiment = final_intra_day_sentiment.drop(columns=['compound', 'timestamp'])

# --- PART 3: COMBINE AND SAVE ---
final_df = pd.merge(final_intra_day_sentiment, daily_agg, on=['ticker', 'date'], how='left')
final_df.fillna(0, inplace=True)

final_df.to_csv(OUT_CSV, index=False)
print(f"Saved corrected cumulative sentiment features -> {OUT_CSV}")