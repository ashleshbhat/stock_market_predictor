#!python3

import requests
import json
import wordlist
import os
import csv
import datetime
import time
import re
import sentiment

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

def compile_news_score(number_of_weeks):
    date_end = datetime.datetime(2018,5,20)
    filename_csv = "AAPL/NYT/newsinfo.csv"

    header = ['date begin','date end','score','number of words','pos_simple','neg_simple']
    s = sentiment.SentimentAnalysis()
        
    with open(filename_csv, 'w',newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
        writer.writerow(header)
        
    for loop in range (0, number_of_weeks):
        date_begin = date_end - datetime.timedelta(days = 6)
        date_end_parsed = str(date_end).replace("-","")[0:8]
        date_begin_parsed = str(date_begin).replace("-","")[0:8]
        
        pos_words = wordlist.get_word_list_pos()
        neg_words = wordlist.get_word_list_neg()

        print("working on the week "+str(date_begin)+" to "+str(date_end))
        filename_txt = "AAPL/NYT/"+date_begin_parsed+"_"+date_end_parsed+".txt"
        pos = 0
        neg = 0
        with open(filename_txt, 'r',encoding='utf-8') as myfile:
            data=myfile.read().replace('\n', '')
            data = re.sub(r'[^a-zA-Z ]+', '', data).replace("xexx","")
            # data = re.sub(r'\W+', '', data)  
            score = s.score(data)
            for word in pos_words:
                if (word) in (data.upper()):
                    pos = pos +1
                
            for word in neg_words:
                if word in data.upper():
                    neg = neg +1

            length =  len(data.split())
            row_data = [date_begin,date_end,score,length,pos,neg]
            with open(filename_csv, 'a',newline='') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
                writer.writerow(row_data)   
        
        date_end = date_end - datetime.timedelta(days = 7)


# get_NYT_multiple_weeks(1000)

compile_news_score(1500)