import pandas as pd

# Load AAPL (from today) and TSLA+GOOGL (from this run)
aapl_df   = pd.read_csv("data/01_RawNewsHeadlines/newsapi_headlines.csv")
other_df  = pd.read_csv("data/01_RawNewsHeadlines/newsapi_headlines_TSLA_GOOGL.csv")

# Concatenate, dedupe, and sort
master_df = pd.concat([aapl_df, other_df], ignore_index=True)
master_df.drop_duplicates(subset=["ticker","publishedAt","title"], inplace=True)
master_df["publishedAt"] = pd.to_datetime(master_df["publishedAt"])
master_df.sort_values(["ticker","publishedAt"], inplace=True)

# Save
master_df.to_csv("data/01_RawNewsHeadlines/newsapi_headlines_all3.csv", index=False)
print(f"Master file has {len(master_df)} headlines for AAPL, TSLA, GOOGL.")
