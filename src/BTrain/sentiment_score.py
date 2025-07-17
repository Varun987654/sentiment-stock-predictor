import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. Load the cleaned headlines
df = pd.read_csv('data/01_RawNewsHeadlines/newsapi_headlines_vader_ready.csv')

# 2. Initialize VADER and compute scores
sia = SentimentIntensityAnalyzer()
scores = df['vader_ready'].apply(sia.polarity_scores)
sent_df = pd.DataFrame(list(scores))

# 3. Combine and save
result = pd.concat([df, sent_df], axis=1)
result.to_csv('data/01_RawNewsHeadlines/newsapi_headlines_with_sentiment.csv', index=False)

print(f"Saved {len(result)} rows with sentiment scores to newsapi_headlines_with_sentiment.csv")
