
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv('notebooks/raw_analyst.ipynb')

# Descriptive Statistics
## Obtain basic statistics for textual lengths
df['headline_length'] = df['headline'].apply(len)
print(df['headline_length'].describe())

## Count the number of articles per publisher
publisher_counts = df['publisher'].value_counts()
print(publisher_counts)

## Analyze the publication dates
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.day_name()
day_counts = df['day_of_week'].value_counts()
print(day_counts)

# Text Analysis
## Perform sentiment analysis on headlines
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Categorize sentiment
df['sentiment_category'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Count sentiment categories
sentiment_counts = df['sentiment_category'].value_counts()
print("Sentiment distribution:")
print(sentiment_counts)
## Identify common keywords or phrases
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['headline'])
word_freq = pd.DataFrame({'word': vectorizer.get_feature_names_out(), 'frequency': np.asarray(X.sum(axis=0)).ravel().tolist()})
print(word_freq.nlargest(columns='frequency', n=10))

# Time Series Analysis
## Publication frequency over time
df['date'].value_counts().plot(kind='line')

## Analysis of publishing times
df['hour'] = df['date'].dt.hour
hour_counts = df['hour'].value_counts()
print(hour_counts)

# Publisher Analysis
## Which publishers contribute most to the news feed
print(publisher_counts)

## Identify unique domains if email addresses are used as publisher names
df['email_domain'] = df['publisher'].apply(lambda x: x.split('@')[1] if '@' in x else np.nan)
domain_counts = df['email_domain'].value_counts()
print(domain_counts)
