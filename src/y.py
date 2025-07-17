import pandas as pd

df = pd.read_csv('data/06_Features/merged_features_split.csv', parse_dates=['bin_end_time'])
# Take the first ticker’s times, sort, compute differences
times = df[df['ticker']=='AAPL']['bin_end_time'].sort_values()
deltas = times.diff().dropna().dt.total_seconds() / 60  # minutes
print("Most common bin size (minutes):", deltas.mode().iloc[0])
