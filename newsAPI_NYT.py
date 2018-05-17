#!python3

import requests
import json
import wordlist
import os
import csv
import datetime
import time


def get_NYT_data(querry ,begin_date,end_date):
    url = ('https://api.nytimes.com/svc/search/v2/articlesearch.json?'
    '&api-key=3db8810edc4847fb895660fffe514214'
    '&q='+querry+
    'fq=Technology'
    '&begin_date='+begin_date+
    '&end_date='+end_date+
    '&fl=snippet,abstract,lead_paragraph'
    )
    print(url)
    response = requests.get(url)
    data = (response.json())
    
    response = data['response']
    docs = response['docs']

    #print(response)
    for news in docs:
        if 'snippet' in news:
            snippet = news['snippet']
        if 'abstract' in news:
            abstract = news['abstract']
        if 'lead_paragraph' in news:    
            lead_paragraph = news['lead_paragraph']
        print(snippet)
        print(abstract)
        print()
    #for news in data[]

get_NYT_data("apple","20170312","20170313")
