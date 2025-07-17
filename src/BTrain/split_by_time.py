import pandas as pd
import os

# Paths
INPUT  = os.path.join('data', '06_Features', 'merged_features_indexed.csv')
OUTPUT = os.path.join('data', '06_Features', 'merged_features_split.csv')

# Load indexed data (with MultiIndex: ticker, bin_end_time)
df = pd.read_csv(INPUT, index_col=['ticker', 'bin_end_time'], parse_dates=['bin_end_time'])

# Add a split column
df['split'] = None

# For each ticker, split by time
for ticker in df.index.get_level_values('ticker').unique():
    df_ticker = df.loc[ticker]
    n = len(df_ticker)

    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    df.loc[(ticker, df_ticker.index[:train_end]), 'split'] = 'train'
    df.loc[(ticker, df_ticker.index[train_end:val_end]), 'split'] = 'val'
    df.loc[(ticker, df_ticker.index[val_end:]), 'split'] = 'test'

# Final check
print(df['split'].value_counts())

# Save result
df.to_csv(OUTPUT)
print("✅ Saved time-based split file to:", OUTPUT)
