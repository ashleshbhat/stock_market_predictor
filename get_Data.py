#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==================================
#Get Data set fr    om the Alpha Vantage api 

# 
#  ==================================
# importing libraries
import time
import requests
import csv
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal




# ==================================
# Block to get Applr intrday data for last 15 days 1min tick
# response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&interval=1min&outputsize=full&symbol=AAPL&apikey=KJLE898BN5KOBVS6&datatype=csv")
# with open("C:/Users/user/Documents/GitHub/stock_market_predictor/apple_intraday_15days_1min.csv","w", newline= '') as file:     
#     file.writelines(response.text)

# # Block to get Apple daily data from IPO in csv format 
# response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=AAPL&apikey=KJLE898BN5KOBVS6&datatype=csv")

# with open("C:/Users/user/Documents/GitHub/stock_market_predictor/apple_daily_full.csv","w", newline= '') as file:     
#     file.writelines(response.text)

## csv format timestamp,open,high,low,close,volume
with open("C:/Users/user/Documents/GitHub/stock_market_predictor/apple_daily_small.csv","r", newline= '') as file:
    reader = csv.reader(file)
    data = list(reader)

#print(data)

data_T = np.array(data).T                                   #transpose the data to access easier
data_T_price_low = data_T[3]                                    #open price
data_T_price_low = data_T_price_low.astype(float)                   #convert sring to float
                                
data_T_price_high = data_T[2]                                    #open price
data_T_price_high = data_T_price_high.astype(float)                   #convert sring to float

data_T_price_open = data_T[1]                                    #open price
data_T_price_open = data_T_price_open.astype(float)                   #convert sring to float

data_T_price_close = data_T[4]                                    #open price
data_T_price_close = data_T_price_close.astype(float)                   #convert sring to float


data_T_price_high_diff = np.concatenate((np.zeros(1,).astype(float),data_T_price_high),axis=0) - \
np.concatenate((data_T_price_high,np.zeros(1,).astype(float)),axis=0)


print(data_T_price_high_diff)
data_T_price_high_diff = np.delete(np.delete(data_T_price_high_diff,len(data_T_price_high)),0)
print(data_T_price_high_diff)

data_T_price_high_diff2 = np.concatenate((np.zeros(1,).astype(float),data_T_price_high_diff),axis=0) - \
np.concatenate((data_T_price_high_diff,np.zeros(1,).astype(float)),axis=0)
data_T_price_high_diff2 = np.delete(np.delete(data_T_price_high_diff2,len(data_T_price_high_diff)),0)

print(data_T_price_high_diff.shape)

t = np.arange(1,len(data_T_price_high),1)
#peaks_price_high_pos = signal.find_peaks_cwt(data_T_price_high,t,min_length=1,gap_thresh= 5)
maxi = signal.argrelmax(data_T_price_high)
mini = signal.argrelmin(data_T_price_low)
print(maxi)


#peaks_price_low_neg = signal.find_peaks_cwt(data_T_price_low,t,noise_perc=0.1,min_length=1)
#print(peaks_price_high_pos)

plt.subplot(2,2,1)

t = np.arange(0.0,len(data_T_price_high), 1)
plt.plot(t,data_T_price_low,'r',data_T_price_high, 'g',ls='',marker='+')
plt.xlabel('date')
plt.ylabel('high low price')
plt.title('charts')
plt.grid(True)

plt.plot(t,data_T_price_open,'b',data_T_price_close, 'r',ls='--')
plt.xlabel('date')
plt.ylabel('high low price')
plt.title('charts')
plt.grid(True)


for peaks in maxi:
    plt.plot(peaks,data_T_price_high[peaks],'bs')

for peaks in mini:
    plt.plot(peaks,data_T_price_low[peaks],'rs')


t2 = np.arange(0,len(data_T_price_high_diff),1)
plt.subplot(2,2,2)
plt.plot(t2,data_T_price_high_diff)

t3= np.arange(0,len(data_T_price_high_diff2),1)
plt.subplot(2,2,4)
plt.plot(t3,data_T_price_high_diff2)




plt.show()  




