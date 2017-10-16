import tweepy
from textblob import TextBlob
import csv
from statistics import mean

# Step 1 - Authenticate
consumer_key = 'Xc4e6EvOAfVvY7EOFRiGmY9g8'
consumer_secret = 'naswUlXRdUmY6Jy2oocNUBoEXj7mG5GoNTE8h2bQMOCE4Yu4dO'

access_token = '13721712-lE607ZkYpEYOy6LYBITxvAoOsxKVK54i8MsV08Ov6'
access_token_secret = 'wODpSKOIEbAFGmuqmhv7vV8UUZQ6lVVYjNogMVIFuGh3r'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Step 3 - Retrieve Tweets
public_tweets = api.search('Trump', count=100)


# CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
# and label each one as either 'positive' or 'negative', depending on the sentiment
# You can decide the sentiment polarity threshold yourself
sentiments = []
FILENAME = 'tweets_labeled.csv'
with open(FILENAME, 'w', encoding="utf-8") as file:
    tweets_writer = csv.writer(file)

    for tweet in public_tweets:
        print(tweet.text)

        # Step 4 Perform Sentiment Analysis on Tweets
        analysis = TextBlob(tweet.text)
        print(analysis.sentiment)
        print("")

        sentiment = 'positive' if analysis.sentiment.polarity > 0 else 'negative'
        sentiments.append(analysis.sentiment.polarity)
        row = [sentiment, tweet.text]
        tweets_writer.writerow(row)

print(mean(sentiments))
