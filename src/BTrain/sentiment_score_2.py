import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. Load the cleaned headlines
df = pd.read_csv('data/01_RawNewsHeadlines/newsapi_headlines(2)_vader_ready.csv')

# 2. Drop any rows where 'vader_ready' is NaN
df = df.dropna(subset=['vader_ready'])

# 3. Initialize VADER and compute scores
sia = SentimentIntensityAnalyzer()
scores = df['vader_ready'].apply(sia.polarity_scores)
sent_df = pd.DataFrame(list(scores))

# 4. Combine and save
result = pd.concat([df, sent_df], axis=1)
result.to_csv('data/01_RawNewsHeadlines/newsapi_headlines_with_sentiment(2).csv', index=False)

print(f"Saved {len(result)} rows with sentiment scores to newsapi_headlines_with_sentiment(2).csv")
