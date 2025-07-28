# src/Corrected_BTrain/build_master_dataset_corrected.py

import os
import pandas as pd
import numpy as np

# --- CONFIG ---
INITIAL_FEATURES_DIR = "data/04_Initial_Features/"
SENTIMENT_FILE = "data/05_Binned_sentiments_corrected/sentiment_features_corrected.csv"
MACRO_FILE = "data/03_Macros/macro_gold_brent.csv"
OUT_FILE = "data/06_Features_corrected/merged_features_split_corrected.csv"
TICKERS = ["AAPL", "GOOGL", "TSLA"]
START_DATE = "2025-04-23"
END_DATE = "2025-06-17"

# --- 1. LOAD CORRECTED DATA ---
print("Loading corrected initial features and sentiment data...")
all_initial_features = []
for ticker in TICKERS:
    df = pd.read_csv(f"{INITIAL_FEATURES_DIR}/price_{ticker}.csv", parse_dates=['bin_end_time'])
    all_initial_features.append(df)
main_df = pd.concat(all_initial_features)

sentiment_df = pd.read_csv(SENTIMENT_FILE, parse_dates=['bin_end_time'])

# --- 2. MERGE DATA SOURCES ---
print("Merging data sources...")
main_df['bin_end_time'] = pd.to_datetime(main_df['bin_end_time'], utc=True)
sentiment_df['bin_end_time'] = pd.to_datetime(sentiment_df['bin_end_time'], utc=True)
main_df['date'] = main_df['bin_end_time'].dt.date

df = pd.merge(main_df, sentiment_df, on=['ticker', 'bin_end_time', 'date'], how='left')
df = df.sort_values(by=['ticker', 'bin_end_time']).reset_index(drop=True)

# Add Macro Features
# FIX: Changed 'Date' to 'date' to match the CSV header
macro_df = pd.read_csv(MACRO_FILE, parse_dates=['date'])
macro_df.columns = ['date', 'gold_price', 'brent_oil']
macro_df['date'] = pd.to_datetime(macro_df['date']).dt.date
df = pd.merge(df, macro_df, on='date', how='left')

# --- 3. CREATE FINAL FEATURES & TARGET ---
print("Creating final features and target variable...")
df['day_of_week'] = df['bin_end_time'].dt.dayofweek
df['bin_slot_number'] = df.groupby(df['bin_end_time'].dt.date).cumcount() + 1
df['is_opening_bin'] = (df['bin_slot_number'] == 1).astype(int)
df['is_closing_bin'] = (df['bin_slot_number'] == 13).astype(int)

df['lagged_after_close_compound_mean'] = df.groupby('ticker')['mean_after_close'].shift(13).fillna(0)
df['lagged_pre_open_compound_mean'] = df.groupby('ticker')['mean_pre_open'].shift(13).fillna(0)
df['lagged_trading_compound_mean'] = df.groupby('ticker')['trading_compound_mean'].shift(1).fillna(0)

df['target_return'] = df.groupby('ticker')['bin_return'].shift(-1)

# --- 4. CLEANUP AND RENAME COLUMNS ---
print("Renaming columns and selecting final feature set...")
df.rename(columns={
    'lagged_return': 'lagged_return',
    'lagged_volume_change': 'volume_change',
    'lagged_bin_range': 'bin_range',
    'trading_compound_count': 'trading_compound_count',
    'trading_compound_mean': 'trading_compound_mean',
    'count_pre_open': 'pre_open_compound_count',
    'mean_pre_open': 'pre_open_compound_mean',
    'count_after_close': 'after_close_compound_count',
    'mean_after_close': 'after_close_compound_mean',
}, inplace=True)

final_columns = [
    'ticker', 'bin_end_time', 'open', 'high', 'low', 'close', 'volume',
    'bin_return', 'volume_change', 'bin_range', 'date', 'session',
    'after_close_compound_count', 'pre_open_compound_count', 'trading_compound_count',
    'after_close_compound_max', 'pre_open_compound_max', 'trading_compound_max',
    'after_close_compound_mean', 'pre_open_compound_mean', 'trading_compound_mean',
    'after_close_compound_min', 'pre_open_compound_min', 'trading_compound_min',
    'after_close_compound_std', 'pre_open_compound_std', 'trading_compound_std',
    'gold_price', 'brent_oil', 'day_of_week', 'bin_slot_number',
    'is_opening_bin', 'is_closing_bin', 'lagged_return',
    'lagged_after_close_compound_mean', 'lagged_pre_open_compound_mean',
    'lagged_trading_compound_mean'
]

for col in final_columns:
    if col not in df.columns:
        df[col] = 0

df['bin_return'] = df['target_return']
final_df = df[final_columns].copy()

# --- 5. FINAL SPLIT AND SAVE ---
print("Splitting data and saving final file...")
train_end = pd.to_datetime("2025-05-30").date()
val_end = pd.to_datetime("2025-06-06").date()
final_df['date'] = pd.to_datetime(final_df['date']).dt.date

final_df['split'] = 'test'
final_df.loc[final_df['date'] <= train_end, 'split'] = 'train'
final_df.loc[(final_df['date'] > train_end) & (final_df['date'] <= val_end), 'split'] = 'val'

final_df.dropna(subset=['bin_return'], inplace=True)

final_df.to_csv(OUT_FILE, index=False)
print(f"✔ Successfully created the final corrected dataset: {OUT_FILE}")