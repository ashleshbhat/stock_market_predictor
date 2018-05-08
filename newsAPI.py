#!python3

import requests
import json
import wordlist
import os
import csv
import datetime
import time

def get_news_data(days=5,stock = 'AAPL',keyword = 'Apple'):
    """
    get_news data gets all top daily news using google news api using the keyword 
    parses then and counts the number of positive and negative words used
    and writes to a csv
    you can specify the number of days from today, in past to search average time is 5sec/day
    and specify the keyword use + between multiple keywords
    use the stock to specify where to store the file
    """
    pos_words = wordlist.get_word_list_pos()
    neg_words = wordlist.get_word_list_neg()
    sources = wordlist.get_news_list()
    
    date_end = datetime.date.today()
    date_begin = date_end - datetime.timedelta(days = 1)

    

    header = ['date','positive','negative']

    filename = filename = "data/"+stock+"/news_"+keyword+".csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w',newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
        writer.writerow(header)

    for days in range(0,days):
        data_to_write = []
        pos =0
        neg =0
        print(date_begin)
        for sources in sources:
            url1 = ('https://newsapi.org/v2/everything?'
            'q='+keyword+'&'
            'from='+str(date_begin)+'T00:00:00&'
            'sources='+
                sources+
                '&'
            'to='+str(date_end)+'T00:00:00&'
            'sortBy=popularity&'
            'apiKey=b543043c6ded4cb59b9e1795d20ece02')#d68a6a0610df45188b1f61f78cc1f54c')
            # print(url1) 
            
            response = requests.get(url1)

            data = (response.json())
            #print(data)
            for data in data['articles']:
                title = str(data['title']).upper()
                body = str(data['description']).upper()
                
                for word in pos_words:
                    if word in (title):
                        pos = pos +1
                    if word in (body):
                        pos = pos +1
                
                for word in neg_words:
                    if word in title:
                        neg = neg +1
                    if word in body:
                        neg = neg +1

        data_to_write.append(date_end)            
        data_to_write.append(neg)
        data_to_write.append(pos) 

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a',newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
            writer.writerow(data_to_write)
        
        date_end = date_end - datetime.timedelta(days = 1)
        date_begin = date_end - datetime.timedelta(days = 1)

# get_news_data()