#!python3

import requests
import json
import wordlist
import os
import csv
import datetime
import time


def get_NYT_data(querry,begin_date,end_date):
    url = ('https://api.nytimes.com/svc/search/v2/articlesearch.json?'
    '&api-key=3db8810edc4847fb895660fffe514214'
    '&q='+querry+
    'fq=Technology'
    '&begin_date='+begin_date+
    '&end_date='+end_date+
    '&fl=snippet,abstract,lead_paragraph'
    )
    #print(url)
    response = requests.get(url)
    data = (response.json())
    
    response = data['response']
    docs = response['docs']

    #print(response)
    output = ""
    news_count= 0
    for news in docs:
        if 'snippet' in news:
            snippet = news['snippet']
            output = output + snippet
            
        if 'abstract' in news:
            abstract = news['abstract']
            output = output + abstract
            
        if 'lead_paragraph' in news:    
            lead_paragraph = news['lead_paragraph']
            output = output + lead_paragraph
        
        news_count = news_count + 1
            
    return output,news_count
        


def get_NYT_multiple_weeks(number_of_weeks):
    date_end = datetime.datetime(2018,5,20)
    for loop in range (0, number_of_weeks):
        date_begin = date_end - datetime.timedelta(days = 6)
        date_end_parsed = str(date_end).replace("-","")[0:8]
        date_begin_parsed = str(date_begin).replace("-","")[0:8]
        print("working on the week "+str(date_begin)+" to "+str(date_end))
        out,count = get_NYT_data("apple iphone,apple ipad,apple,apple ipod,macbook,macintosh",date_begin_parsed,date_end_parsed)
        filename = "AAPL/NYT/"+date_begin_parsed+"_"+date_end_parsed+".txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        out_file = open(filename,"w")
        out_file.write(str(out.encode("utf-8")))
        out_file.close()
        
        date_end = date_end - datetime.timedelta(days = 7)
        time.sleep(0.5)

get_NYT_multiple_weeks(1000)