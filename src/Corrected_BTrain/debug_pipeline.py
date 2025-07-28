# src/Corrected_BTrain/debug_pipeline.py

import os
import pandas as pd
import numpy as np

# --- CONFIG ---
PRICE_DIR = "data/03_StockPrices"
NEWS_CSV = "data/01_RawNewsHeadlines/FINAL.csv"
TICKERS = ["AAPL", "GOOGL", "TSLA"]

print("--- Starting Pipeline Debug Trace ---")

# --- Load all raw data once ---
all_prices = []
for ticker in TICKERS:
    price_df = pd.read_csv(f"{PRICE_DIR}/price_{ticker}.csv", parse_dates=['bin_end_time'])
    price_df['ticker'] = ticker
    all_prices.append(price_df)
df_raw_prices = pd.concat(all_prices)
df_raw_prices.columns = df_raw_prices.columns.str.lower()
df_raw_prices['bin_end_time'] = pd.to_datetime(df_raw_prices['bin_end_time'], utc=True)

news_df = pd.read_csv(NEWS_CSV, parse_dates=["publishedAt"])
news_df = news_df.dropna(subset=["ticker", "publishedAt"])
news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"], utc=True)

# --- Process each ticker individually and print status ---
for ticker in TICKERS:
    print(f"\n--- Tracing Ticker: {ticker} ---")
    
    # 1. Start with the raw price data for this ticker
    df = df_raw_prices[df_raw_prices['ticker'] == ticker].copy()
    df = df.sort_values(by='bin_end_time').reset_index(drop=True)
    print(f"1. Initial raw price rows: {len(df)}")

    # 2. Engineer price-based features
    df['bin_return_raw'] = (df['close'] / df['open']) - 1
    df['prev_close'] = df['close'].shift(1)
    # Check rows after a simple shift
    print(f"2. Rows with valid 'prev_close': {df['prev_close'].notna().sum()}")

    # 3. Engineer longest-lookback technical indicator (MACD)
    exp1 = df['prev_close'].ewm(span=12, adjust=False).mean()
    exp2 = df['prev_close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    # Check rows after 26-period EWM
    print(f"3. Rows with valid 'MACD': {df['macd'].notna().sum()}")

    # 4. Process and merge sentiment data
    df['date'] = df['bin_end_time'].dt.date
    news_df['date'] = news_df['publishedAt'].dt.date
    
    # Cumulative trading sentiment
    def session_for(ts: pd.Timestamp) -> str:
        m = ts.hour * 60 + ts.minute
        if 13*60 + 30 <= m < 20*60: return "trading"
        return "other"
    
    ticker_news = news_df[news_df['ticker'] == ticker]
    ticker_news["session"] = ticker_news["publishedAt"].apply(session_for)
    trading_news = ticker_news[ticker_news['session'] == 'trading'].sort_values('publishedAt')
    
    if not trading_news.empty:
        trading_news['cum_count'] = trading_news.groupby('date').cumcount() + 1
        df = pd.merge_asof(
            df.sort_values('bin_end_time'),
            trading_news[['publishedAt', 'cum_count']],
            left_on='bin_end_time', right_on='publishedAt', direction='backward'
        )
        df['cum_count'].ffill(inplace=True)
    else:
        df['cum_count'] = 0 # If no news, count is 0
    
    # Check rows after sentiment merge
    print(f"4. Rows with valid sentiment count: {df['cum_count'].notna().sum()}")

    # 5. Simulate the final dropna()
    final_rows = df.dropna(subset=['prev_close', 'macd', 'cum_count'])
    print(f"5. Final rows after dropping all NaNs: {len(final_rows)}")
    if not final_rows.empty:
        print(f"   -> First valid date would be: {final_rows['bin_end_time'].dt.date.iloc[0]}")
    else:
        print("   -> No valid data would remain.")