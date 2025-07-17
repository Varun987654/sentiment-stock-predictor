import os
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import pandas as pd
from logger import log

# Load API key
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    log.error("NEWSAPI_KEY is not loaded")
    exit(1)

# Tomorrow’s tickers
TICKERS = {
    "TSLA": "Tesla",
    "GOOGL": "Google"
}

def fetch_30_days_per_ticker():
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=29)  # inclusive = 30 days
    all_rows = []

    for ticker, name in TICKERS.items():
        log.info(f"Fetching 30 days of headlines for {ticker}")
        for offset in range(30):
            day = start_date + timedelta(days=offset)
            params = {
                "q":        f"{ticker} OR \"{name}\"",
                "from":     day.isoformat(),
                "to":       day.isoformat(),
                "language": "en",
                "pageSize": 100,
                "page":     1,
                "sortBy":   "publishedAt",
                "apiKey":   NEWSAPI_KEY
            }
            resp = requests.get("https://newsapi.org/v2/everything", params=params)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            log.info(f"  • {day}: {len(articles)} articles")
            for a in articles:
                all_rows.append({
                    "ticker":      ticker,
                    "date":        day.isoformat(),
                    "publishedAt": a["publishedAt"],
                    "source":      a["source"]["name"],
                    "title":       a["title"]
                })

    # Save combined DataFrame
    df = pd.DataFrame(all_rows)
    os.makedirs("data/01_RawNewsHeadlines", exist_ok=True)
    path = "data/01_RawNewsHeadlines/newsapi_headlines_TSLA_GOOGL.csv"
    df.to_csv(path, index=False)
    log.info(f"Saved {len(df)} total headlines (30 days × {len(TICKERS)} tickers) to {path}")

if __name__ == "__main__":
    fetch_30_days_per_ticker()
