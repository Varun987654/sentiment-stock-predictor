import os
import requests
import pandas as pd
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ─── CONFIG ────────────────────────────────────────────────
API_KEY     = os.getenv("NEWSAPI_KEY")  # ensure your key is set in env
TICKERS     = ["AAPL", "GOOGL", "TSLA"]
MISSING_DAY = "2025-05-22"
PAGE_SIZE   = 100                           # get up to 100 headlines
RAW_DIR     = "data/01_RawNewsHeadlines"
VADER_READY = f"{RAW_DIR}/newsapi_headlines(2)_vader_ready.csv"
BINS_CSV    = "data/05_Binned_sentiment/news_sentiment_30min_bins.csv"

# ─── FETCH RAW HEADLINES ──────────────────────────────────
all_new = []
for ticker in TICKERS:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        ticker,
        "from":     MISSING_DAY,
        "to":       MISSING_DAY,
        "sortBy":   "publishedAt",
        "pageSize": PAGE_SIZE,
        "apiKey":   API_KEY
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    for art in articles:
        all_new.append({
            "ticker":      ticker,
            "date":        MISSING_DAY,
            "publishedAt": art["publishedAt"],
            "source":      art["source"]["name"],
            "title":       art["title"]
        })
# save interim raw
os.makedirs(RAW_DIR, exist_ok=True)
new_raw_path = f"{RAW_DIR}/newsapi_headlines_2025-05-22.csv"
pd.DataFrame(all_new).to_csv(new_raw_path, index=False)
print(f"Fetched {len(all_new)} new headlines for {MISSING_DAY}")

# ─── CLEAN + VADER READY ──────────────────────────────────
import re
raw_df = pd.DataFrame(all_new)
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'\s+', ' ', text).strip()
raw_df['vader_ready'] = raw_df['title'].astype(str).apply(clean_text)
clean_path = f"{RAW_DIR}/newsapi_headlines_2025-05-22_vader_ready.csv"
raw_df.to_csv(clean_path, index=False)
print(f"Cleaned headlines saved to {clean_path}")

# ─── VADER SCORING ────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()
for col in ['neg','neu','pos','compound']:
    raw_df[col] = raw_df['vader_ready'].apply(lambda t: analyzer.polarity_scores(t)[col])
scored_path = f"{RAW_DIR}/newsapi_headlines_2025-05-22_with_sentiment.csv"
raw_df.to_csv(scored_path, index=False)
print(f"Scored headlines saved to {scored_path}")

# ─── UPDATE BINS ──────────────────────────────────────────
bins_df = pd.read_csv(BINS_CSV, parse_dates=['bin_end_time'])
# bin MISSING_DAY headlines into 30-min slots
raw_df['bin_end_time'] = pd.to_datetime(raw_df['publishedAt']).dt.floor('30min') + pd.Timedelta(minutes=30)
# aggregate per bin
agg = raw_df.groupby('bin_end_time')['compound'].agg(['count','mean','std','max','min']).reset_index()
agg.columns = ['bin_end_time','count_new','mean_new','std_new','max_new','min_new']
# merge replacing same-day bins
merged = bins_df.merge(agg, on='bin_end_time', how='left')
# replace only those bins on MISSING_DAY
mask = merged['bin_end_time'].dt.date == datetime.fromisoformat(MISSING_DAY).date()
for col in ['count','mean_compound','std_compound','max_compound','min_compound']:
    merged.loc[mask, col] = merged.loc[mask, f"{col.split('_')[0]}_new"]
# drop helper cols
merged = merged[bins_df.columns]
updated_bins_path = BINS_CSV.replace('.csv','_updated.csv')
merged.to_csv(updated_bins_path, index=False)
print(f"Updated bins saved to {updated_bins_path}")
