import pandas as pd
import os

# File paths
INPUT  = os.path.join('data', '06_Features', 'merged_features.csv')
OUTPUT = os.path.join('data', '06_Features', 'merged_features_indexed.csv')

# Load data
df = pd.read_csv(INPUT)

# Convert to datetime
df['bin_end_time'] = pd.to_datetime(df['bin_end_time'], utc=True)
df['date'] = pd.to_datetime(df['date'])

# Sort by ticker and bin_end_time
df = df.sort_values(['ticker', 'bin_end_time'])

# Drop duplicates if any (same ticker and same timestamp)
df = df.drop_duplicates(subset=['ticker', 'bin_end_time'], keep='first')

# Set multi-index (optional but helpful for slicing)
df = df.set_index(['ticker', 'bin_end_time'])

# Save to output
df.to_csv(OUTPUT)
print("✅ Saved indexed file to:", OUTPUT)
