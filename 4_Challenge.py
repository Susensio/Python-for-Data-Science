import tweepy
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# Step 1 - Insert your API keys
consumer_key = 'Xc4e6EvOAfVvY7EOFRiGmY9g8'
consumer_secret = 'naswUlXRdUmY6Jy2oocNUBoEXj7mG5GoNTE8h2bQMOCE4Yu4dO'
access_token = '13721712-lE607ZkYpEYOy6LYBITxvAoOsxKVK54i8MsV08Ov6'
access_token_secret = 'wODpSKOIEbAFGmuqmhv7vV8UUZQ6lVVYjNogMVIFuGh3r'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Step 2 - Search for your company name on Twitter
public_tweets = api.search('apple', count=100)


# Step 3 - Define a threshold for each sentiment to classify each
# as positive or negative. If the majority of tweets you've collected are positive
# then use your neural network to predict a future price
sentiments = []
for tweet in public_tweets:
    analysis = TextBlob(tweet.text)
    sentiments.append(analysis.sentiment.polarity)

sentiment = sum(sentiments)
print(sentiment)


def get_data(filename):
    df = pd.read_csv(filename, parse_dates=['Date'], usecols=['Date', 'Close'])
    df = pd.read_csv(filename, usecols=['Close'])

    return df


# Step 5 reference your CSV file here
df = get_data('data/stock/aapl.csv')
# dataset = df[:,'Close'].values
dataset = df.values.astype('float32')

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# reshape into X=t and Y=t+1
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# Step 6 In this function, build your neural network model using Keras, train it, then have it predict the price
# on a given day. We'll later print the price out to terminal.
np.random.seed(7)

# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(100, input_dim=look_back, activation='relu'))
model.add(Dense(100, input_dim=look_back, activation='relu'))
model.add(Dense(50, input_dim=look_back, activation='relu'))
model.add(Dense(25, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, trainScore**0.5))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, testScore ** 0.5))

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# def predict_prices(dates, prices, x):
#     pass


# predicted_price = predict_prices(dates, prices, 29)
# print(predicted_price)
