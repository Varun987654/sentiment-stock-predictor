import pandas as pd

# Paths to your two scored‑headline CSVs
old_csv = "data/01_RawNewsHeadlines/newsapi_headlines_final_with_sentiments.csv"
new_csv = "data/01_RawNewsHeadlines/newsapi_headlines_2025-05-22_with_sentiment.csv"

# 1. Load both files (with publishedAt parsed as datetime)
df_old = pd.read_csv(old_csv, parse_dates=["publishedAt"])
df_new = pd.read_csv(new_csv, parse_dates=["publishedAt"])

# 2. Concatenate into one DataFrame
df_merged = pd.concat([df_old, df_new], ignore_index=True)

# 3. (Optional) Sort by ticker and timestamp
df_merged = df_merged.sort_values(["ticker", "publishedAt"]).reset_index(drop=True)

# 4. Save to a new CSV
output_path = "data/01_RawNewsHeadlines/FINAL.csv"
df_merged.to_csv(output_path, index=False)

print(f"Merged file saved to {output_path} with {len(df_merged)} rows.")
