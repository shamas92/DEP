import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_data = pd.read_csv('twitter_training.csv')
validation_data = pd.read_csv('twitter_validation.csv')

# Preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        text = ''
    text = text.lower()  # convert to lowercase
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'#\w+', '', text)  # remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

train_data['cleaned_text'] = train_data.iloc[:, -1].apply(preprocess_text)
validation_data['cleaned_text'] = validation_data.iloc[:, -1].apply(preprocess_text)

# Sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

train_data['sentiment'] = train_data['cleaned_text'].apply(get_sentiment)
validation_data['sentiment'] = validation_data['cleaned_text'].apply(get_sentiment)

# Save the DataFrames to CSV for later review
train_data.to_csv('train_data_sentiment_analysis.csv', index=False)
validation_data.to_csv('validation_data_sentiment_analysis.csv', index=False)

# Visualization of sentiment distribution in training data
plt.figure(figsize=(10, 5))
sns.countplot(data=train_data, x='sentiment')
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('train_data_sentiment_distribution.png')
plt.show()

# Visualization of sentiment distribution in validation data
plt.figure(figsize=(10, 5))
sns.countplot(data=validation_data, x='sentiment')
plt.title('Sentiment Distribution in Validation Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('validation_data_sentiment_distribution.png')
plt.show()

# Post count by media company 
plt.figure(figsize=(10, 5))
sns.countplot(y=train_data.iloc[:, 1], order=train_data.iloc[:, 1].value_counts().index)
plt.title('Post Count by Media Company in Training Data')
plt.xlabel('Count')
plt.ylabel('Media Company')
plt.savefig('train_data_post_count_by_media_company.png')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(y=validation_data.iloc[:, 1], order=validation_data.iloc[:, 1].value_counts().index)
plt.title('Post Count by Media Company in Validation Data')
plt.xlabel('Count')
plt.ylabel('Media Company')
plt.savefig('validation_data_post_count_by_media_company.png')
plt.show()
