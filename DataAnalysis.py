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
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# import numerical_algo


def load_stock_data_intraday(company,date):
    date_filename = "data/"+company+"/date/"+date+".csv"
    return pd.read_csv(date_filename, parse_dates=['date'])

def load_stock_data_daily(company="AAPL", output_size="compact"):
    date_filename = "data/"+company+"/"+company+"_daily_"+output_size+".csv"
    return pd.read_csv(date_filename, parse_dates=['date'])

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

def training(k=5):
    # print (stockData.info())
    stockData = pd.read_csv("AAPL/stockData.csv")
    stockData.dropna()

    X = stockData.drop("Action", axis=1)
    X = X.drop("date", axis=1)
    Y = stockData["Action"].copy()

    Xtrain, Xtest = train_test_split(X, test_size=0.2, random_state=k)
    Ytrain, Ytest = train_test_split(Y, test_size=0.2, random_state=k)
    # print (train)
    # print(test)
    
    model = RandomForestClassifier()
    # train the model
    model.fit(Xtrain, Ytrain)
    # test the model
    Yp = model.predict(Xtest)

    print(Yp)
    print(Ytest)

    CM = confusion_matrix (Yp, Ytest)
    print (CM)


def process_stock(stock="AAPL", _print=False):
    stockData = load_stock_data_daily(stock, "compact")
    stockData.set_index('date',inplace=True)

    # get first degree difference
    # Abs
    stockData['firstDiffAbs'] = stockData["close"].diff(periods=-1)
    # percentage
    stockData['firstDiff_%'] = stockData.firstDiffAbs/stockData.close * 100
    
    # plot_bargraph(stockData.index, stockData['firstDiffAbs'], "Diff1Abs", "blue")
    # plot_bargraph(stockData.index, stockData['firstDiff_%'], "Diff1_%", "red")

    # stockData.info()
    # print (stockData)
    appleNews = load_news("AAPL")
    appleNews.set_index('date', inplace=True)
    # create rating column and normalize values
    appleNews['rating'] = (appleNews.positive - appleNews.negative)/(appleNews.positive + abs(appleNews.negative))
    appleNews.info()
    if(_print): 
        plot_bargraph(appleNews.index, appleNews.rating, "newsClassification", "weekly")
    # print(appleNews) 

    # add newsinfo as colum to stockdata
    stockData['newsRating'] = appleNews['rating']
    # stockData.dropna()
    stockData.drop("open", axis=1, inplace=True)
    stockData.drop("high", axis=1, inplace=True)
    stockData.drop("low", axis=1, inplace=True)
    # stockData.drop("open", axis=1, inplace=True)
    stockData.info()
    stockData.corr() # see and plot correlations 
    if(_print):
        pd.plotting.scatter_matrix(stockData[["firstDiffAbs","newsRating"]])
        pd.plotting.scatter_matrix(stockData[["close","newsRating"]])
    plt.show()

    stockData.to_csv("AAPL/stockData.csv")

    return stockData
    # print(appleNews['2018-05-08 '])
# =======================================
# data = process_stock(_print=False)
training()