import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the dataset
file_path = "../data/raw_analyst_ratings.csv"
df= pd.read_csv(file_path)
df

print("Column names and data types:")
print(df.dtypes)

obj = (df.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (df.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (df.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))

# Descriptive Statistics
## Obtain basic statistics for textual lengths
df['headline_length'] = df['headline'].apply(len)
print(df['headline_length'].describe())

df['headline_length'] = df['headline'].apply(len)

# Basic statistics
mean_length = df['headline_length'].mean()
median_length = df['headline_length'].median()
std_dev = df['headline_length'].std()
range_min = df['headline_length'].min()
range_max = df['headline_length'].max()

print(f"Mean headline length: {mean_length:.2f} characters")
print(f"Median headline length: {median_length:.2f} characters")
print(f"Standard deviation: {std_dev:.2f}")
print(f"Range: {range_min} - {range_max} characters")

## Count the number of articles per publisher
article_counts = df['publisher'].value_counts()
print("Articles per publisher:")
print(article_counts)

## Analyze the publication dates
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.day_name()
day_counts = df['day_of_week'].value_counts()

print(day_counts)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek 

day_counts = df.groupby('day_of_week')['headline'].count()

# Plot the results
plt.figure(figsize=(8, 5))
sns.barplot(x=day_counts.index, y=day_counts.values)
plt.xlabel('Day of the Week')
plt.ylabel('Number of Articles')
plt.title('News Frequency by Day of the Week')
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()


# Create a histogram
plt.figure(figsize=(8, 5))
sns.histplot(day_counts, bins=7, kde=False, color='skyblue', edgecolor='black')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Articles')
plt.title('News Frequency by Day of the Week')
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()

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
# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Group articles by date and count the number of publications
daily_publications = df.groupby(df['date'].dt.date)['headline'].count()

# Plot the publication frequency over time
plt.figure(figsize=(10, 6))
daily_publications.plot(kind='line', marker='o', color='b')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.title('Publication Frequency of Financial News Articles')
plt.grid(True)
plt.show()

# Extract the hour of publication
df['hour_of_publication'] = df['date'].dt.hour

# Group articles by hour and count the number of publications
hourly_publications = df.groupby('hour_of_publication')['headline'].count()

# Plot the results
plt.figure(figsize=(8, 5))
hourly_publications.plot(kind='bar', color='g')
plt.xlabel('Hour of Publication')
plt.ylabel('Number of Articles')
plt.title('Publication Times of Financial News Articles')
plt.xticks(rotation=0)
plt.show()


## Analysis of publishing times
df['hour'] = df['date'].dt.hour
hour_counts = df['hour'].value_counts()
print(hour_counts)

# Extract unique domains from email addresses (if applicable)
df['publisher_domain'] = df['publisher'].str.split('@').str[-1]

# Count the number of articles per publisher
publisher_counts = df['publisher'].value_counts()

# Print top publishers and their article counts
print("Top Publishers:")
print(publisher_counts.head(10))

# Print unique publisher domains (if applicable)
unique_domains = df['publisher_domain'].nunique()
print(f"Unique Publisher Domains: {unique_domains}")

## Identify unique domains if email addresses are used as publisher names
df['email_domain'] = df['publisher'].apply(lambda x: x.split('@')[1] if '@' in x else np.nan)
domain_counts = df['email_domain'].value_counts()
print(domain_counts)