# src/Corrected_train/Phase1_build_advanced_features.py

import os
import pandas as pd
import numpy as np

# --- CONFIG ---
PRICE_DIR = "data/03_StockPrices"
NEWS_CSV = "data/01_RawNewsHeadlines/FINAL.csv"
MACRO_FILE = "data/03_Macros/macro_gold_brent.csv"
OUT_FILE = "data/06_Features_corrected/advanced_features.csv"
TICKERS = ["AAPL", "GOOGL", "TSLA"]

print("--- Starting Advanced Feature Engineering Process ---")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# --- STEP 1: LOAD ALL RAW DATA ---
print("Step 1: Loading all raw data...")
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

# --- STEP 2: PROCESS & MERGE SENTIMENT DATA (with Rolling Windows) ---
print("Step 2: Engineering advanced sentiment features...")
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

# Daily pre-open and (lagged) after-close (remains the same)
daily_agg = news_df[news_df['session'] != 'trading'].groupby(["ticker", "date", "session"])["compound"].agg(['count', 'mean']).unstack().fillna(0)
daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]
after_cols = [c for c in daily_agg.columns if 'after_close' in c]
if after_cols:
    daily_agg = daily_agg.reset_index()
    daily_agg['date'] = pd.to_datetime(daily_agg['date'])
    daily_agg[after_cols] = daily_agg.groupby('ticker')[after_cols].shift(1).fillna(0)
df = pd.merge(df, daily_agg, on=['ticker', 'date'], how='left')

# ADVANCED: Rolling window trading sentiment
trading_news = news_df[news_df['session'] == 'trading'].set_index('publishedAt').sort_values('publishedAt')
# FIX: Use 'min' instead of 'T' for modern pandas syntax
roll_60m = trading_news.groupby('ticker')['compound'].rolling('60min').agg(['mean', 'count']).rename(columns=lambda x: f"trading_compound_{x}_60m")
roll_180m = trading_news.groupby('ticker')['compound'].rolling('180min').agg(['mean', 'count']).rename(columns=lambda x: f"trading_compound_{x}_180m")
rolling_aggs = pd.concat([roll_60m, roll_180m], axis=1).reset_index()

# FIX: Sort the rolling aggregates by the timestamp before merging
rolling_aggs = rolling_aggs.sort_values('publishedAt')

df = pd.merge_asof(
    df.sort_values('bin_end_time'),
    rolling_aggs.rename(columns={'publishedAt': 'bin_end_time'}),
    on='bin_end_time', by='ticker', direction='backward'
)

# --- STEP 3: MERGE MACRO & CREATE ADVANCED FEATURES ---
print("Step 3: Engineering advanced technical and interaction features...")
macro_df = pd.read_csv(MACRO_FILE, parse_dates=['date'])
df['date'] = pd.to_datetime(df['date'])
macro_df['date'] = pd.to_datetime(macro_df['date'])
df = pd.merge(df, macro_df, on='date', how='left')

all_ticker_dfs_final = []
for ticker, group in df.groupby('ticker'):
    ticker_df = group.copy()

    # Target and Lagged Price-based Features
    ticker_df['bin_return_raw'] = (ticker_df['close'] / ticker_df['open']) - 1
    ticker_df['target_return'] = ticker_df['bin_return_raw'].shift(-1)
    ticker_df['prev_close'] = ticker_df['close'].shift(1)
    ticker_df['prev_high'] = ticker_df['high'].shift(1)
    ticker_df['prev_low'] = ticker_df['low'].shift(1)
    ticker_df['lagged_return'] = ticker_df['bin_return_raw'].shift(1)

    # ADVANCED: Add ATR (Average True Range) for volatility
    tr1 = ticker_df['high'] - ticker_df['low']
    tr2 = np.abs(ticker_df['high'] - ticker_df['prev_close'])
    tr3 = np.abs(ticker_df['low'] - ticker_df['prev_close'])
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    ticker_df['atr_14'] = tr.rolling(window=14).mean().shift(1) # Lagged ATR

    # ADVANCED: Interaction Feature
    ticker_df['sentiment_x_volatility'] = ticker_df['trading_compound_mean_60m'] * ticker_df['atr_14']

    # Time-based features
    ticker_df['day_of_week'] = ticker_df['bin_end_time'].dt.dayofweek
    ticker_df['bin_slot_number'] = ticker_df.groupby('date').cumcount() + 1

    all_ticker_dfs_final.append(ticker_df)

final_df = pd.concat(all_ticker_dfs_final)

# --- STEP 4: CLEANUP, SPLIT, AND SAVE ---
print("Step 4: Cleaning, splitting, and saving the final advanced feature file...")
final_feature_columns = [
    'ticker', 'bin_end_time', 'open', 'volume', 'target_return', 'split',
    'lagged_return', 'prev_close', 'prev_high', 'prev_low',
    'count_after_close', 'mean_after_close', 'count_pre_open', 'mean_pre_open',
    'trading_compound_mean_60m', 'trading_compound_count_60m',
    'trading_compound_mean_180m', 'trading_compound_count_180m',
    'atr_14', 'sentiment_x_volatility',
    'gold_price', 'brent_oil',
    'day_of_week', 'bin_slot_number'
]

final_df.ffill(inplace=True)
for col in final_feature_columns:
    if col not in final_df.columns:
        final_df[col] = 0

train_end = pd.to_datetime("2025-05-30", utc=True)
val_end = pd.to_datetime("2025-06-06", utc=True)
final_df['split'] = 'test'
final_df.loc[final_df['bin_end_time'] <= train_end, 'split'] = 'train'
final_df.loc[(final_df['bin_end_time'] > train_end) & (final_df['bin_end_time'] <= val_end), 'split'] = 'val'

final_df = final_df[final_feature_columns].copy()
final_df.dropna(inplace=True)

final_df.to_csv(OUT_FILE, index=False)
print(f"✔✔✔ ADVANCED FEATURE SET COMPLETE. The file is saved at: {OUT_FILE}")