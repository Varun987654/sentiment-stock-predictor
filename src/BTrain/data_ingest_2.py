import os
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import pandas as pd
from logger import log

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    log.error("NEWSAPI_KEY is not loaded")
    exit(1)

TICKERS = {
    "AAPL": "Apple",
    "GOOGL": "Google",
    "TSLA": "Tesla"
}
START_DATE = datetime.fromisoformat("2025-05-23").date()
END_DATE   = datetime.fromisoformat("2025-06-17").date()

def fetch_date_range_headlines():
    all_rows = []
    total_days = (END_DATE - START_DATE).days + 1

    for ticker, name in TICKERS.items():
        log.info(f"Fetching headlines for {ticker} from {START_DATE} to {END_DATE}")
        for offset in range(total_days):
            day = START_DATE + timedelta(days=offset)
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

    os.makedirs("data/01_RawNewsHeadlines", exist_ok=True)
    out_path = "data/01_RawNewsHeadlines/newsapi_headlines(2).csv"
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    log.info(f"Saved {len(all_rows)} headlines to {out_path}")

if __name__ == "__main__":
    fetch_date_range_headlines()
