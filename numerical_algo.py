
#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# importing libraries
import time
import requests
import csv
import get_data
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal
import os


def return_stats_intra(company,date, plot=0):
    """send comapny code name and the date and this function returns the data, the first difference of high price and low price
    the normalised first differnce of high pric ande low price.
    it also returns 3 differnet peak anysis with 1,3,5 peak analysis
    use it like this [a,b,c,d,e,f,g,h] =return_stats_intra("GOOGL","2018-04-17",plot=1)
    a is full data read ofcource first 30 and last 30 mins neglected
    b is the first difference of high values
    c is first difference of low values
    d is the first dif of high values normaliesed in %
    e is the first dif of low values normaliesed in %
    f is max function with 1 consecutive max
    g is max function with 2 consecutive max
    h is max function with 5 consecutive max
    you can snd plot = 1 to plot a chart 
    please use date in correct format and which is availabe in database
    please use the company name which is standard and available in database
        """
    date_filename = "data/"+company+"/date/"+date+".csv"
    if(os.path.isfile(date_filename)):
        with open(date_filename, 'r',newline='') as myfile: 
            reader = csv.reader(myfile)
            data = list(reader)

        print(len(data))
        if(len(data)>350):
            data = data[30:len(data)-30]
            
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

            data_T_price_low_diff = np.concatenate((np.zeros(1,).astype(float),data_T_price_low),axis=0) - \
            np.concatenate((data_T_price_low,np.zeros(1,).astype(float)),axis=0)

            data_T_price_high_diff_norm = ((np.concatenate((np.zeros(1,).astype(float),data_T_price_high),axis=0) - \
            np.concatenate((data_T_price_high,np.zeros(1,).astype(float)),axis=0)) / \
            np.concatenate((data_T_price_high,np.zeros(1,).astype(float)),axis=0))*100 #percentage change

            data_T_price_low_diff_norm = ((np.concatenate((np.zeros(1,).astype(float),data_T_price_low),axis=0) - \
            np.concatenate((data_T_price_low,np.zeros(1,).astype(float)),axis=0)) / \
            np.concatenate((data_T_price_low,np.zeros(1,).astype(float)),axis=0))*100 #percentage change

            data_T_price_high_diff = data_T_price_high_diff[1:len(data_T_price_high_diff)-1]
            data_T_price_low_diff = data_T_price_low_diff[1:len(data_T_price_low_diff)-1]
            data_T_price_high_diff_norm = data_T_price_high_diff_norm[1:len(data_T_price_high_diff)-1]
            data_T_price_low_diff_norm = data_T_price_low_diff_norm[1:len(data_T_price_low_diff)-1]
           
            t = np.arange(1,len(data_T_price_high),1)
            
            maxi_1 = signal.find_peaks_cwt(data_T_price_low,t,noise_perc=0.1,min_length=1)
            maxi_2 = signal.find_peaks_cwt(data_T_price_low,t,noise_perc=0.1,min_length=2)
            maxi_5 = signal.find_peaks_cwt(data_T_price_low,t,noise_perc=0.1,min_length=5)
            if(plot ==1):
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
                
                plt.grid(True)
                
                t2 = np.arange(0,len(data_T_price_high_diff),1)
                plt.subplot(2,2,2)
                plt.plot(t2,data_T_price_high_diff)
                plt.plot(t2,data_T_price_low_diff)
                plt.xlabel('time in mins')
                plt.ylabel('first difference')
                plt.grid(True)

                for peaks in maxi_1:
                    plt.plot(peaks,data_T_price_high_diff[peaks],'g',ls='',marker='+')

                for peaks in maxi_2:
                    plt.plot(peaks,data_T_price_high_diff[peaks],'r',ls='',marker='+')

                for peaks in maxi_5:
                    plt.plot(peaks,data_T_price_high_diff[peaks],'b',ls='',marker='+')


                t3 = np.arange(0,len(data_T_price_high_diff_norm),1)
                plt.subplot(2,2,4)
                plt.plot(t3,data_T_price_high_diff_norm)
                plt.plot(t3,data_T_price_low_diff_norm)
                plt.xlabel('time in mins')
                plt.ylabel('normalised first difference in %')
                plt.grid(True)
                plt.show()

            return data_T,data_T_price_high_diff,data_T_price_high_diff_norm,data_T_price_low_diff,data_T_price_low_diff_norm,maxi_1,maxi_1,maxi_5
        else:
            print("data on the date "+date+" is too small")
        return
    else:
        print("NO info available for the company "+ company +" for the date "+date)
        return 


# how to use
""" 
input arguments ("Company Name", "Date", 0 or 1 for plotting)
"""
# [a,b,c,d,e,f,g,h] =return_stats_intra("AAPL","2018-04-20",plot=1)



