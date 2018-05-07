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
import numerical_algo


def load_stock_data_intraday(company,date):
    date_filename = "data/"+company+"/date/"+date+".csv"
    return pd.read_csv(date_filename, parse_dates=['date'])

def load_stock_data_daily(company="AAPL", output_size="compact"):
    date_filename = "data/"+company+"/"+company+"_daily_"+output_size+".csv"
    return pd.read_csv(date_filename, parse_dates=['date'])

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

    [total,diff_hi,diff_low,perc_high,perc_low,f,g,h] =numerical_algo.return_stats_intra("AAPL","2018-04-20",plot=1)

    print (total)
# ==================================
def plot_bargraph(x, _data, _label, _color="red"):
    #plot data
    fig, ax = plt.subplots(figsize=(15,7))

    ax.bar(x, _data, color=_color, label=_label)
    plt.legend(loc='best')

    #set ticks every week
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.show()

def process_stock(stock="AAPL"):
    stockData = load_stock_data_daily(stock, "compact")
    stockData.info()
    stockData.set_index('date',inplace=True)

    # get first degree difference
    # Abs
    stockData['firstDiffAbs'] = stockData["close"].diff() 
    # percentage
    stockData['firstDiff_%'] = stockData.firstDiffAbs/stockData.close * 100
    
    plot_bargraph(stockData.index, stockData['firstDiffAbs'], "Diff1Abs", "blue")
    plot_bargraph(stockData.index, stockData['firstDiff_%'], "Diff1_%", "red")

    stockData.info()
    # print (stockData)

# =======================================
process_stock()