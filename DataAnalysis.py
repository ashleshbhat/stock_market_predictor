#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================
# 
# Using pandas library for data processing
# 
# =============================


# importing libraries
import time
import requests
import csv
import math, time
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from scipy import signal
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import pandas_datareader.data as web
import datetime
import technical_indicators as TI
# neural netowork libraries
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

# import numerical_algo

# ====== input parameters =============
seq_len = 22
d = 0.2
# shape = [10, seq_len, 1] # feature, window, output
neurons = [128, 128, 32, 1]
epochs = 300
# =====================================

def load_stock_data_intraday(company,date):
    data_filename = "data/"+company+"/date/"+date+".csv"
    return pd.read_csv(data_filename, parse_dates=['timestamp'])

def load_stock_data_daily(company="AAPL", output_size="compact"):
    data_filename = "data/"+company+"/"+company+"_daily_"+output_size+".csv"
    return pd.read_csv(data_filename, parse_dates=['timestamp'])

def load_stock_data_weekly(company="AAPL"):
    data_filename = company+"/weekly_adjusted_"+company+".csv"
    return pd.read_csv(data_filename, parse_dates=['timestamp'])

def load_news(company="AAPL"):
    return pd.read_csv(company+"/NYT/newsinfo.csv", parse_dates=['date begin'])

def load_file(file="", dateString=""):
    return pd.read_csv(file, parse_dates=[dateString])

def playground():
    stock_data = load_stock_data_daily("AAPL")
    stock_data.head()
    stock_data.info()
    stock_data.describe()
    stock_data.hist()
    # show the graph
    plt.show()

    stock_data.corr() # see and plot correlations 
    pd.plotting.scatter_matrix(stock_data[["close","volume"]])
    plt.show()
    stock_data.info()

    # [total,diff_hi,diff_low,perc_high,perc_low,f,g,h] =numerical_algo.return_stats_intra("AAPL","2018-04-20",plot=1)

    # print (total)
# ==================================
def readStockWeb(_stock="AAPL",freq="weekly"):
    start   = datetime.datetime(2000,1,1)
    end     = datetime.datetime.today()
    # read from morningstar
    dat = web.DataReader(_stock,"morningstar",start, end)
    # print(dat)

# function to normalize the stock data
def normalizeData(df):      
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['adjusted close'] = min_max_scaler.fit_transform(df['adjusted close'].values.reshape(-1,1))
    return df
def plot_bargraph(x, _data, _label="no_label", frame="weekly",_color="red"):
    #plot data
    fig, ax = plt.subplots(figsize=(15,7))

    ax.bar(x, _data, color=_color, label=_label)
    plt.legend(loc='best')

    #set ticks per selection
    if(frame == "monthly"):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    elif (frame == "yearly"):
        ax.xaxis.set_major_locator(mdates.YearLocator())
    elif(frame == "weekly"):
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.show()

def plot_result(stock_name, predicted, actual):
    plt.plot(predicted, color='red', label='Prediction')
    plt.plot(actual,color='blue', label='Actual')
    plt.legend(loc='best')
    plt.title('The test result for {}'.format(stock_name))
    plt.xlabel('Days')
    plt.ylabel('Adjusted Close')
    plt.show()
    
def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)
    row = round(0.9 * result.shape[0]) # 90% split
    
    train = result[:int(row), :] # 90% date
    X_train = train[:, :-1] # all data until day m
    y_train = train[:, -1][:,-1] # day m + 1 adjusted close price
    
    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1] 

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

    return [X_train, y_train, X_test, y_test]

# create neural network model
def create_nn_model(layers, neurons, d):
    model = Sequential()
    # [4, 22, 1]
    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))
    # model = load_model('my_LSTM_stock_model1000.h5')
    # adam = keras.optimizers.Adam(decay=0.2)
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# create another neural network
def create_nn_model2():
    model = Sequential()

    model.add(LSTM(
        input_dim=1,
        output_dim=50,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print ('compilation time : ', time.time() - start)
    return model

def training_classification(file, k=5):
    # print (stockData.info())
    stockData = load_file(file, "timestamp")
    stockData.set_index('timestamp',inplace=True)
    stockData = stockData.dropna()
    # drop the columns open, high and low
    stockData.drop("open", axis=1, inplace=True)
    stockData.drop("high", axis=1, inplace=True)
    stockData.drop("low", axis=1, inplace=True)
    stockData.drop("dividend amount", axis=1, inplace=True)
    stockData.drop("firstDiffAbs", axis=1, inplace=True)
    stockData.drop("firstDiff_%", axis=1, inplace=True)
    stockData.drop("volume", axis=1, inplace=True)
    stockData.drop("ROC_2", axis=1, inplace=True)

    # save features to X
    X = stockData.drop("Action", axis=1)
    # X = X.drop("timestamp", axis=1)
    # save target to Y - in our case the Action to do (Put/Call)
    Y = stockData["Action"].copy()

    # split into training and test set
    Xtrain, Xtest = train_test_split(X, test_size=0.2, random_state=k)
    Ytrain, Ytest = train_test_split(Y, test_size=0.2, random_state=k)
    # print (train)
    # print(test)
    Xtrain.info()
    Xtest.info()
    
    model = RandomForestClassifier()
    # train the model
    model.fit(Xtrain, Ytrain)
    # test the model
    Yp = model.predict(Xtest)

    # create confusion matrix
    CM = confusion_matrix (Yp, Ytest)
    print (CM)
    accur = metrics.accuracy_score(Ytest,Yp)
    precision = metrics.precision_score(Ytest,Yp, average="macro")
    print ("Accuracy: ",accur,"\nPrecisionScore: ", precision)


def training_neuralnet(file):
    # load stock
    stockData = load_file(file, "timestamp")
    stockData.set_index('timestamp',inplace=True)
    stockData = stockData.dropna()
    # ===== neural network for prediction ========
    # create neural net

    stockData.drop("Action", axis=1, inplace=True) # drop Action column because not needed for regression
    stockData.drop("volume", axis=1, inplace=True)
    stockData.drop("dividend amount", axis=1, inplace=True)
    # normalize the data
    # stockData = normalizeData(stockData)
    X_train, y_train, X_test, y_test = load_data(stockData, seq_len)
    print(X_train.shape[0], X_train.shape[1], X_train.shape[2])


    shape = [10, seq_len, 1] # feature, window, output
    model_nn = create_nn_model(shape, neurons, d)
    model_nn.fit(
        X_train,
        y_train,
        batch_size=512,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )

    model_score(model_nn, X_train, y_train, X_test, y_test)
    p = percentage_difference(model_nn, X_test, y_test)
    # print(p)

    plot_result("Apple",p, y_test)

    return model_nn

# calculate score of the training and the test set
def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]

def percentage_difference(model, X_test, y_test):
    percentage_diff=[]

    p = model.predict(X_test)
    for u in range(len(y_test)): # for each data index in test data
        pr = p[u][0] # pr = prediction on day u

        percentage_diff.append((pr-y_test[u]/pr)*100)
    return p

def process_stock(_stock="AAPL", _print=False, _type="compact",freq="daily" ,_loadnews=False, _normalize=False):
    if (freq == "daily"):
        stockData = load_stock_data_daily(_stock, _type)
    elif (freq == "weekly"):
        stockData = load_stock_data_weekly(company=_stock)

    stockData.set_index('timestamp',inplace=True)

    # drop close column because only adjusted close is interesting because of the stock splits
    stockData.drop('close', axis=1, inplace=True)

    # get first degree difference and create new column
    # Abssolute
    stockData['firstDiffAbs'] = stockData["adjusted close"].diff(periods=-1)
    # percentage
    stockData['firstDiff_%'] = stockData.firstDiffAbs/stockData["adjusted close"] * 100
    
    # add ROC-1
    stockData = TI.rate_of_change(stockData,2)


    # normalize data
    if _normalize:        
        stockData = normalizeData(stockData)

    # plt.plot(stockData['adjusted close'], color='blue')
    # plt.show()

    # save processed file with new columns
    stockData.to_csv(_stock+"/weekly_adjusted_"+_stock+"_processed.csv")
    # stockData.to_csv("AAPL/stockData.csv")
    # stockData.to_csv("AAPL/StockFull.csv")
    
    print ("processed the stock with Symbol: "+_stock)

    return stockData
# =======================================
def addNews(file="AAPL/weekly_adjusted_AAPL_processed.csv", _loadnews=True,_print=False):
    stockData = pd.read_csv(file, parse_dates=['timestamp'])
    stockData.set_index('timestamp', inplace=True)
    # check if news information shall be loaded for the stock
    if(_loadnews):
        # load apple news nyt and set index
        News = load_file("AAPL/NYT/newsinfo.csv","date begin")
        News.set_index('date begin',inplace=True)

        if(_print): 
            # print news classification
            plot_bargraph(News.index, News['score'],"News Score", "yearly")

        # add news rating as colum to stockdata
        stockData['newsRating'] = News['score']
        # stockData += News.pos_simple

        stockData.info()
        if(_print):
            stockData.corr() # see and plot correlations 
            pd.plotting.scatter_matrix(stockData[["firstDiffAbs","newsRating"]])
            pd.plotting.scatter_matrix(stockData[["adjusted close","newsRating"]])
            plt.show()
    stockData.to_csv("AAPL/weekly_adjusted_AAPL_processed_final.csv")

# data = process_stock(_print=False,_loadnews=True)
# process_stock(freq="weekly",_normalize=False)
# training(file="AAPL/stockData.csv")
# training(file="AAPL/StockFull.csv")
training_classification(file="AAPL/weekly_adjusted_AAPL_corr.csv")
# nn = training_neuralnet("AAPL/weekly_adjusted_AAPL_corr.csv")


# addNews()
