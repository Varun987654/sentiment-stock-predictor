import pandas as pd
import os

# 1. File paths
file1 = "data/01_RawNewsHeadlines/newsapi_headlines_with_sentiment.csv"
file2 = "data/01_RawNewsHeadlines/newsapi_headlines_with_sentiment(2).csv"
output_file = "data/01_RawNewsHeadlines/newsapi_headlines_final_with_sentiments.csv"

# 2. Load both files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 3. Strip extra spaces from column names (just in case)
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# 4. Convert publishedAt to datetime
df1['publishedAt'] = pd.to_datetime(df1['publishedAt'], errors='coerce')
df2['publishedAt'] = pd.to_datetime(df2['publishedAt'], errors='coerce')

# 5. Concatenate
merged_df = pd.concat([df1, df2], ignore_index=True)

# 6. Sort by ticker and publishedAt
merged_df = merged_df.sort_values(by=['ticker', 'publishedAt']).reset_index(drop=True)

# 7. Save the merged file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
merged_df.to_csv(output_file, index=False)

print(f"Merged file saved to {output_file} with {len(merged_df)} rows.")
