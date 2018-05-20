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
# import get_data
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
# import numerical_algo


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
    return pd.read_csv(company+"/news_Apple.csv", parse_dates=['date'])

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
    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    return df
def plot_bargraph(x, _data, _label="no_label", frame="weekly",_color="red"):
    #plot data
    fig, ax = plt.subplots(figsize=(15,7))

    ax.bar(x, _data, color=_color, label=_label)
    plt.legend(loc='best')

    #set ticks per selection
    if(frame == "monthly"):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.show()

def training(file, k=5):
    # print (stockData.info())
    stockData = pd.read_csv(file)
    stockData.dropna()
    # drop the columns open, high and low
    # stockData.drop("open", axis=1, inplace=True)
    # stockData.drop("high", axis=1, inplace=True)
    # stockData.drop("low", axis=1, inplace=True)
    stockData.drop("dividend amount", axis=1, inplace=True)

    # save features to X
    X = stockData.drop("Action", axis=1)
    X = X.drop("timestamp", axis=1)
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

    # check if news information shall be loaded for the stock
    if(_loadnews):
        appleNews = load_news(_stock)
        appleNews.set_index('date', inplace=True)
        # create rating column and normalize values
        appleNews['rating'] = (appleNews.positive - appleNews.negative)/(appleNews.positive + abs(appleNews.negative))
        appleNews.info()
        if(_print): 
            # print news classification
            plot_bargraph(appleNews.index, appleNews.rating, "newsClassification", "weekly")

        # add news rating as colum to stockdata
        stockData['newsRating'] = appleNews['rating']

        stockData.info()
        stockData.corr() # see and plot correlations 
        if(_print):
            pd.plotting.scatter_matrix(stockData[["firstDiffAbs","newsRating"]])
            pd.plotting.scatter_matrix(stockData[["adjusted close","newsRating"]])
        plt.show()

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
# data = process_stock(_print=False,_loadnews=True)
process_stock(freq="weekly",_normalize=False)
# training(file="AAPL/stockData.csv")
# training(file="AAPL/StockFull.csv")
training(file="AAPL/weekly_adjusted_AAPL_corr.csv")