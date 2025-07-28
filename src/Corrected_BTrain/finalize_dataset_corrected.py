# src/Corrected_BTrain/build_final_dataset_full.py

import os
import pandas as pd
import numpy as np

# --- CONFIG ---
PRICE_DIR = "data/03_StockPrices"
NEWS_CSV = "data/01_RawNewsHeadlines/FINAL.csv"
MACRO_FILE = "data/03_Macros/macro_gold_brent.csv"
OUT_FILE = "data/06_Features_corrected/merged_features_split_corrected.csv"
TICKERS = ["AAPL", "GOOGL", "TSLA"]
START_DATE = "2025-04-23"
END_DATE = "2025-06-17"

print("--- Starting Final Data Build Process ---")
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
df['date'] = df['bin_end_time'].dt.date


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
news_df['date'] = news_df['publishedAt'].dt.date

# Daily pre-open and (lagged) after-close
daily_agg = news_df[news_df['session'] != 'trading'].groupby(["ticker", "date", "session"])["compound"].agg(['count', 'mean']).unstack().fillna(0)
daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]
after_cols = [c for c in daily_agg.columns if 'after_close' in c]
if after_cols:
    daily_agg = daily_agg.reset_index()
    daily_agg[after_cols] = daily_agg.groupby('ticker')[after_cols].shift(1).fillna(0)
df['date'] = pd.to_datetime(df['date'])
daily_agg['date'] = pd.to_datetime(daily_agg['date'])
df = pd.merge(df, daily_agg, on=['ticker', 'date'], how='left')

# Cumulative intra-day trading sentiment
trading_news = news_df[news_df['session'] == 'trading'].sort_values('publishedAt')
trading_news['cum_sum'] = trading_news.groupby(['ticker', 'date'])['compound'].cumsum()
trading_news['cum_count'] = trading_news.groupby(['ticker', 'date']).cumcount() + 1
trading_news['trading_compound_mean'] = trading_news['cum_sum'] / trading_news['cum_count']
trading_news.rename(columns={'cum_count': 'trading_compound_count'}, inplace=True)
df = pd.merge_asof(
    df.sort_values('bin_end_time'),
    trading_news[['publishedAt', 'ticker', 'trading_compound_count', 'trading_compound_mean']],
    left_on='bin_end_time', right_on='publishedAt', by='ticker', direction='backward'
).drop(columns='publishedAt')

# --- STEP 3: PROCESS & MERGE MACRO DATA ---
print("Step 3: Merging macro data...")
macro_df = pd.read_csv(MACRO_FILE, parse_dates=['date'])
df['date'] = pd.to_datetime(df['date'])
macro_df['date'] = pd.to_datetime(macro_df['date'])
df = pd.merge(df, macro_df, on='date', how='left')

# --- STEP 4: CREATE FINAL FEATURES & TARGET (PER TICKER) ---
print("Step 4: Engineering all final features and target...")
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
    ticker_df['lagged_volume_change'] = ticker_df['volume'].pct_change().shift(1)
    ticker_df['lagged_bin_range'] = (ticker_df['high'].shift(1) - ticker_df['low'].shift(1))

    # All Lagged Technical Indicators
    ticker_df['sma_10'] = ticker_df['prev_close'].rolling(window=10).mean()
    ticker_df['ema_10'] = ticker_df['prev_close'].ewm(span=10, adjust=False).mean()
    delta = ticker_df['prev_close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    ticker_df['rsi'] = 100 - (100 / (1 + rs))
    exp1 = ticker_df['prev_close'].ewm(span=12, adjust=False).mean()
    exp2 = ticker_df['prev_close'].ewm(span=26, adjust=False).mean()
    ticker_df['macd'] = exp1 - exp2
    ticker_df['macd_signal'] = ticker_df['macd'].ewm(span=9, adjust=False).mean()
    sma_20 = ticker_df['prev_close'].rolling(window=20).mean()
    std_20 = ticker_df['prev_close'].rolling(window=20).std()
    ticker_df['bb_high'] = sma_20 + (std_20 * 2)
    ticker_df['bb_low'] = sma_20 - (std_20 * 2)

    # Time-based features
    ticker_df['day_of_week'] = ticker_df['bin_end_time'].dt.dayofweek
    ticker_df['bin_slot_number'] = ticker_df.groupby('date').cumcount() + 1
    
    all_ticker_dfs_final.append(ticker_df)

final_df = pd.concat(all_ticker_dfs_final)

# --- STEP 5: CLEANUP, SPLIT, AND SAVE ---
print("Step 5: Cleaning, splitting, and saving the final file...")
final_feature_columns = [
    'ticker', 'bin_end_time', 'open', 'volume', 'target_return', 'split',
    'lagged_return', 'lagged_volume_change', 'lagged_bin_range',
    'prev_close', 'prev_high', 'prev_low',
    'sma_10', 'ema_10', 'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low',
    'count_after_close', 'mean_after_close', 'count_pre_open', 'mean_pre_open',
    'trading_compound_count', 'trading_compound_mean', 'gold_price', 'brent_oil',
    'day_of_week', 'bin_slot_number'
]

# Fill NaNs created by merges and shifts then select final columns
final_df.ffill(inplace=True)
for col in final_feature_columns:
    if col not in final_df.columns:
        final_df[col] = 0
        
# Set split
train_end = pd.to_datetime("2025-05-30", utc=True)
val_end = pd.to_datetime("2025-06-06", utc=True)
final_df['split'] = 'test'
final_df.loc[final_df['bin_end_time'] <= train_end, 'split'] = 'train'
final_df.loc[(final_df['bin_end_time'] > train_end) & (final_df['bin_end_time'] <= val_end), 'split'] = 'val'

# Final selection and drop of any remaining NaNs
final_df = final_df[final_feature_columns].copy()
final_df.dropna(inplace=True)

final_df.to_csv(OUT_FILE, index=False)
print(f"✔✔✔ FINAL SCRIPT COMPLETE. The fully corrected dataset with all features is saved at: {OUT_FILE}")