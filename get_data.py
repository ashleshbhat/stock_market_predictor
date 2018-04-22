#!python3
# -*- coding: utf-8 -*-

# ==================================
#Get Data set fr    om the Alpha Vantage api 

# 
#  ==================================
# importing libraries

import requests
import csv
import os
import numpy as np

#=====================================================



def get_stock(stock_name ="AAPL",series="daily",data_type="csv",output_size="compact",interval ="1",print_to_file=1):
    '''stock_name shoud be the standard stock ID    - default AAPL.
    series can be intraday or daily                 - default daily.
    datatype can be csv or json                     - default csv.
    output size compact or full                     - default conmpact.
    interval only for intraday 1,5,15,30,60         - default 1.
    print_to_file - 1 to print 0 to not             - default 1.
    '''
    if output_size not in {"compact","full"}:
        print("Wrong Size. Use compact or full")
        return
    if interval not in {"1","5","15","30","60"}:
        print("wrong interval. Use 1 5 15 30 60")
        return
    if data_type not in {"csv","json"}:
        print("Wrong data type. Use csv or json")
        return

    if(series == "intraday"):
        response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&interval="+interval+"min&outputsize="+output_size+"&symbol="+stock_name+"&apikey=KJLE898BN5KOBVS6&datatype="+data_type)
        filename = "data/"+stock_name+"/"+stock_name+"_"+series+"_"+output_size+"_"+interval+"."+data_type
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        text = response.text.split('\n',1)[-1]
        with open(filename,"w", newline= '') as file:     
            file.writelines(text)
        print("file written to "+filename)
    else:
        if(series == "daily"):
            response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize="+output_size+"&symbol="+stock_name+"&apikey=KJLE898BN5KOBVS6&datatype="+data_type)
            filename = "data/"+stock_name+"/"+stock_name+"_"+series+"_"+output_size+"."+data_type
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            text = response.text.split('\n',1)[-1]
            with open(filename,"w", newline= '') as file:     
                file.writelines(text)
            print("file written to "+filename)
        else:
            print("Wrong series type."+series+" does not exist!"+"your options are intraday or daily")
            return
    return response

#==================================
#example
#res = get_stock("ABEO","daily",data_type="csv")

#print(type(res.text))

def get_all_companies():
    filename = "data/NASDAQ_companies.csv"
    with open(filename,"r",newline= '') as file:
        reader = csv.reader(file, lineterminator = '\n')
        company_data = list(reader)
    
    company_data = np.array(company_data) 
     
    for row in range (0,len(company_data)-1):
        print("working on "+company_data[row][0])
        get_stock(company_data[row][0],"daily",data_type="csv",output_size="full")
        get_stock(company_data[row][0],"intraday",data_type="csv",output_size="full")


get_all_companies()