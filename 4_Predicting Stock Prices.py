import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []


def get_data(filename):
    format = '%d-%b-%y'
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        [next(csvFileReader) for _ in range(230)]
        for i, row in enumerate(csvFileReader):
            dates.append(i)
            prices.append(float(row[1]))
    print("Data uploaded")
    return


def predict_price(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    print("Start fitting models")
    svr_lin.fit(dates, prices)
    print("Linear fitted")
    svr_poly.fit(dates, prices)
    print("Polynomial fitted")
    svr_rbf.fit(dates, prices)
    print("RBF fitted")

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_lin.predict(x)[0], svr_poly.predict(x)[0], svr_rbf.predict(x)[0]


get_data('data/stock/aapl.csv')
print(len(dates))
predicted_price = predict_price(dates, prices, 13)

print(predicted_price)
