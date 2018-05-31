# Computation of News data

The processing of news information is done in the `newsAPI_NYT.py` file:
First we call the function `get_NYT_multiple_weeks(number_of_weeks)` to gather the news through the API as described in [here](https://github.com/ashleshbhat/stock_market_predictor/wiki/Gathering-news-information-through-the-NYT-data-feed).

Then we call the function `compile_news_score(number_of_weeks)` to process the gathered news information and to create numerical data for it which we can later use for our algorithms.

```python
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

```

The output of the news processing function was the file `AAPL/NYT/newsinfo.csv` with the following columns:
> 'date begin','date end','score','number of words','pos_simple','neg_simple'
* The first two columns indicating the start and end date of the week processed. 
* The `score` column containing the score of the sentiment analysis 
* The `number of words` column containing the number of words in the weeks news file.
* The `pos_simple` column containing the number of positive words according to our previous method by manually creating a list with positive words.
* The `neg_simple ` column containing the number of negative words according to our previous method by manually creating a list with negative words.

## External resources for sentiment analysis
In order to compute the sentiment analysis score we made use of an external tool based on the `SentiWordNet 3.0` which is explained [here](https://github.com/ashleshbhat/stock_market_predictor/blob/master/sentiment/ReadMe.md).