# src/Corrected_BTrain/merge_data_corrected.py

import os
import pandas as pd

# --- CONFIG ---
INITIAL_FEATURES_DIR = "data/04_Initial_Features/"
SENTIMENT_FILE = "data/05_Binned_sentiments_corrected/sentiment_features_corrected.csv"
MACRO_FILE = "data/03_Macros/macro_gold_brent.csv"
OUT_FILE = "data/06_Features_corrected/merged_corrected.csv"
TICKERS = ["AAPL", "GOOGL", "TSLA"]

# --- PREP ---
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# 1. Load Corrected Initial Features
print("Loading corrected initial features...")
all_initial_features = []
for ticker in TICKERS:
    file_path = f"{INITIAL_FEATURES_DIR}/price_{ticker}.csv"
    df = pd.read_csv(file_path, parse_dates=['bin_end_time'])
    all_initial_features.append(df)
price_df = pd.concat(all_initial_features)

# 2. Load Corrected Sentiment Features
print("Loading corrected sentiment features...")
sentiment_df = pd.read_csv(SENTIMENT_FILE, parse_dates=['bin_end_time'])

# 3. Load Macro Data
print("Loading macro data...")
macro_df = pd.read_csv(MACRO_FILE, parse_dates=['date'])

# 4. Merge All Data Sources
print("Merging all data sources...")

# --- FIX: Convert all date-related columns to consistent types before merging ---
price_df['bin_end_time'] = pd.to_datetime(price_df['bin_end_time'], utc=True)
sentiment_df['bin_end_time'] = pd.to_datetime(sentiment_df['bin_end_time'], utc=True)

# Create a consistent 'date' column of type datetime64[ns] in both DataFrames
price_df['date'] = pd.to_datetime(price_df['bin_end_time'].dt.date)
sentiment_df['date'] = pd.to_datetime(sentiment_df['date']) # This was the missing conversion
macro_df['date'] = pd.to_datetime(macro_df['date'])

# Merge price features with sentiment features
merged_df = pd.merge(price_df, sentiment_df, on=['ticker', 'bin_end_time', 'date'], how='left')

# Merge with macro features
merged_df = pd.merge(merged_df, macro_df, on='date', how='left')

# Forward-fill macro data for weekends/holidays
merged_df[['gold_price', 'brent_oil']] = merged_df[['gold_price', 'brent_oil']].ffill()

# 5. Add Time-Based Features
print("Adding time-based features...")
merged_df = merged_df.sort_values(by=['ticker', 'bin_end_time']).reset_index(drop=True)
merged_df['day_of_week'] = merged_df['bin_end_time'].dt.dayofweek
merged_df['bin_slot_number'] = merged_df.groupby(['ticker','date']).cumcount() + 1
last_slot = merged_df.groupby(['ticker','date'])['bin_slot_number'].transform('max')
merged_df['is_opening_bin'] = (merged_df['bin_slot_number'] == 1).astype(int)
merged_df['is_closing_bin'] = (merged_df['bin_slot_number'] == last_slot).astype(int)

# 6. Save the intermediate merged file
merged_df.to_csv(OUT_FILE, index=False)
print(f"✔ Successfully created the merged data file: {OUT_FILE}")