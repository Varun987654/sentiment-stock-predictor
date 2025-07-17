import re
import pandas as pd

# 1. Read the raw CSV we just made
df = pd.read_csv("data/01_RawNewsHeadlines/newsapi_headlines(2).csv")

# 2. Define a simple function to clean text for VADER
def clean(text):
    # a) cut out any http:// or https:// links
    text = re.sub(r'http\S+', '', text)
    # b) remove HTML tags like <p> or <br>
    text = re.sub(r'<[^>]+>', '', text)
    # c) change multiple spaces or newlines into one space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Make a new column 'vader_ready' by cleaning the 'title'
df['vader_ready'] = df['title'].astype(str).apply(clean)

# 4. Drop rows where cleaning made an empty string
df = df[df['vader_ready'] != '']

# 5. Drop exact duplicates so we don’t score the same text twice
df = df.drop_duplicates(subset='vader_ready')

# 6. Save the cleaned file for the next step
df.to_csv("data/01_RawNewsHeadlines/newsapi_headlines(2)_vader_ready.csv", index=False)

print(f"Saved {len(df)} cleaned headlines to data/01_RawNewsHeadlines/newsapi_headlines(2)_vader_ready.csv")
