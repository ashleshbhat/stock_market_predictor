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
from scipy import signal
import os
import pandas as pd
import numerical_algo


def load_stock_data_intraday(company,date):
    date_filename = "data/"+company+"/date/"+date+".csv"
    return pd.read_csv(date_filename)

def load_stock_data_daily(company="AAPL", output_size="compact"):
    date_filename = "data/"+company+"/"+company+"_daily_"+output_size+".csv"
    return pd.read_csv(date_filename)

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

def process_stock(stock="AAPL"):
    Apple = load_stock_data_daily(stock)
    Apple.info()
    # get first degree difference
    d1= Apple["close"].diff()
    plt.plot(d1,color="red", label="Diff1")
    plt.legend(loc='best')
    plt.show()

process_stock()