# src/Corrected_BTrain/build_final_dataset_v3.py

import os
import pandas as pd
import numpy as np

# --- CONFIG ---
PRICE_DIR = "data/03_StockPrices"
NEWS_CSV = "data/01_RawNewsHeadlines/FINAL.csv"
MACRO_FILE = "data/03_Macros/macro_gold_brent.csv"
OUT_FILE = "data/06_Features_corrected/merged_features_split_corrected_v3.csv"
TICKERS = ["AAPL", "GOOGL", "TSLA"]
START_DATE = "2025-04-23"
END_DATE = "2025-06-17"

print("--- Starting Final, Corrected Data Build Process ---")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# --- STEP 1: LOAD & PREPARE PRICE DATA ---
print("Step 1: Loading and preparing price data...")
all_prices = []
for ticker in TICKERS:
    price_df = pd.read_csv(f"{PRICE_DIR}/price_{ticker}.csv", parse_dates=['bin_end_time'])
    price_df['ticker'] = ticker
    all_prices.append(price_df)
df = pd.concat(all_prices)
df.columns = df.columns.str.lower()
df['bin_end_time'] = pd.to_datetime(df['bin_end_time'], utc=True)
df = df.sort_values(by=['ticker', 'bin_end_time']).reset_index(drop=True)
df['date'] = pd.to_datetime(df['bin_end_time'].dt.date)

# --- STEP 2: PROCESS & MERGE SENTIMENT DATA ---
print("Step 2: Processing and merging sentiment data...")
news_df = pd.read_csv(NEWS_CSV, parse_dates=["publishedAt"])
news_df = news_df.dropna(subset=["ticker", "publishedAt"])
news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"], utc=True)

def session_for(ts: pd.Timestamp) -> str:
    m = ts.hour * 60 + ts.minute
    if 21*60 <= m < 24*60: return "after_close"
    if 0 <= m < 13*60 + 30: return "pre_open"
    return "trading"
news_df["session"] = news_df["publishedAt"].apply(session_for)
news_df['date'] = pd.to_datetime(news_df['publishedAt'].dt.date)

# Daily pre-open and (lagged) after-close
daily_agg = news_df[news_df['session'] != 'trading'].groupby(["ticker", "date", "session"])["compound"].agg(['count', 'mean', 'std', 'max', 'min']).unstack().fillna(0)
daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]
after_cols = [c for c in daily_agg.columns if 'after_close' in c]
if after_cols:
    daily_agg = daily_agg.reset_index()
    daily_agg['date'] = pd.to_datetime(daily_agg['date'])
    daily_agg[after_cols] = daily_agg.groupby('ticker')[after_cols].shift(1).fillna(0)
df = pd.merge(df, daily_agg, on=['ticker', 'date'], how='left')

# Cumulative intra-day trading sentiment
trading_news = news_df[news_df['session'] == 'trading'].sort_values('publishedAt')
all_trading_aggs = []
for group_keys, group_df in trading_news.groupby(['ticker', 'date']):
    group_df['trading_compound_count'] = group_df['compound'].expanding().count()
    group_df['trading_compound_mean'] = group_df['compound'].expanding().mean()
    group_df['trading_compound_std'] = group_df['compound'].expanding().std().fillna(0)
    group_df['trading_compound_max'] = group_df['compound'].expanding().max()
    group_df['trading_compound_min'] = group_df['compound'].expanding().min()
    all_trading_aggs.append(group_df)
trading_aggs_df = pd.concat(all_trading_aggs)

# --- FIX: Explicitly sort the right-side DataFrame before merge_asof ---
trading_aggs_df = trading_aggs_df.sort_values('publishedAt')

df = pd.merge_asof(
    df.sort_values('bin_end_time'),
    trading_aggs_df[['publishedAt', 'ticker', 'trading_compound_count', 'trading_compound_mean', 'trading_compound_std', 'trading_compound_max', 'trading_compound_min']],
    left_on='bin_end_time', right_on='publishedAt', by='ticker', direction='backward'
).drop(columns='publishedAt')

# --- STEP 3: MERGE MACRO & CREATE FINAL FEATURES ---
print("Step 3: Merging macro and creating final features...")
macro_df = pd.read_csv(MACRO_FILE, parse_dates=['date'])
df['date'] = pd.to_datetime(df['date'])
macro_df['date'] = pd.to_datetime(macro_df['date'])
df = pd.merge(df, macro_df, on='date', how='left')

all_ticker_dfs_final = []
for ticker, group in df.groupby('ticker'):
    ticker_df = group.copy()
    ticker_df['bin_return_raw'] = (ticker_df['close'] / ticker_df['open']) - 1
    ticker_df['target_return'] = ticker_df['bin_return_raw'].shift(-1)
    ticker_df['prev_close'] = ticker_df['close'].shift(1)
    ticker_df['prev_high'] = ticker_df['high'].shift(1)
    ticker_df['prev_low'] = ticker_df['low'].shift(1)
    ticker_df['lagged_return'] = ticker_df['bin_return_raw'].shift(1)
    ticker_df['lagged_volume_change'] = ticker_df['volume'].pct_change().shift(1)
    ticker_df['lagged_bin_range'] = (ticker_df['high'].shift(1) - ticker_df['low'].shift(1))
    ticker_df['sma_10'] = ticker_df['prev_close'].rolling(window=10).mean()
    ticker_df['day_of_week'] = ticker_df['bin_end_time'].dt.dayofweek
    ticker_df['bin_slot_number'] = ticker_df.groupby('date').cumcount() + 1
    all_ticker_dfs_final.append(ticker_df)
final_df = pd.concat(all_ticker_dfs_final)

# --- STEP 4: CLEANUP & SAVE ---
print("Step 4: Cleaning and saving final file...")
final_columns = [
    'ticker', 'bin_end_time', 'open', 'volume', 'target_return', 'split',
    'lagged_return', 'lagged_volume_change', 'lagged_bin_range', 'prev_close', 'prev_high', 'prev_low',
    'sma_10', 'count_after_close', 'mean_after_close', 'std_after_close', 'max_after_close', 'min_after_close',
    'count_pre_open', 'mean_pre_open', 'std_pre_open', 'max_pre_open', 'min_pre_open',
    'trading_compound_count', 'trading_compound_mean', 'trading_compound_std', 'trading_compound_max', 'trading_compound_min',
    'gold_price', 'brent_oil', 'day_of_week', 'bin_slot_number'
]
final_df.ffill(inplace=True)
for col in final_columns:
    if col not in final_df.columns:
        final_df[col] = 0
train_end = pd.to_datetime("2025-05-30", utc=True)
val_end = pd.to_datetime("2025-06-06", utc=True)
final_df['split'] = 'test'
final_df.loc[final_df['bin_end_time'] <= train_end, 'split'] = 'train'
final_df.loc[(final_df['bin_end_time'] > train_end) & (final_df['bin_end_time'] <= val_end), 'split'] = 'val'
final_df = final_df[final_columns].copy()
final_df.dropna(inplace=True)

final_df.to_csv(OUT_FILE, index=False)
print(f"✔✔✔ FINAL SCRIPT COMPLETE. The file is saved at: {OUT_FILE}")