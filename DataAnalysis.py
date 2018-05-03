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


def load_stock_data_intraday(company,date):
    date_filename = "data/"+company+"/date/"+date+".csv"
    return pd.read_csv(date_filename)

def load_stock_data_daily(company="AAPL", output_size="compact"):
    date_filename = "data/"+company+"/"+company+"_daily_"+output_size+".csv"
    return pd.read_csv(date_filename)

stock_data = load_stock_data_daily("AAPL")
stock_data.info()
stock_data.describe()
stock_data.hist()
# show the graph
plt.show()

stock_data.corr() # see and plot correlations 
pd.plotting.scatter_matrix(stock_data[["close","volume"]])
plt.show()
stock_data.info()