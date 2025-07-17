import re
import pandas as pd

# 1. Load your raw CSV from the specified folder
df = pd.read_csv("data/01_RawNewsHeadlines/newsapi_headlines_all3.csv")

# 2. Define a minimal cleaning function for VADER
def clean(text):
    # a) Remove URLs
    text = re.sub(r'http\S+', '', text)
    # b) Remove simple HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # c) Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Apply the cleaner to the 'title' column and store results in 'vader_ready'
df['vader_ready'] = df['title'].astype(str).apply(clean)

# 4. Remove exact duplicates and any rows where cleaning produced an empty string
df = df.drop_duplicates(subset='vader_ready')
df = df[df['vader_ready'] != '']

# 5. Save the cleaned output to a new CSV (you can change this path as needed)
df.to_csv("data/01_RawNewsHeadlines/newsapi_headlines_vader_ready.csv", index=False)

print(f"Saved {len(df)} cleaned headlines to data/01_RawNewsHeadlines/newsapi_headlines_vader_ready.csv")
