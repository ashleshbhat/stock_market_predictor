# stock_market_predictor

## Documentation
Needed libraries:
- requests library      --> `pip3 install requests`
- matplotlib library    --> `pip3 install matplotlib`
- scipy library         --> `pip3 install scipy`
- pandas library        --> `pip3 install pandas`

# stock_market_predictor

## Introduction
This repo is the final project for BIGDATA subject in MASTEAM.
Basically the project is to predict stock exchange price for some comapnies listed in NASDAQ.
## Alpha Vantage API:
(1) Stock Time Series Data: Time Series Data provides realtime and historical equity data in 4 different temporal resolutions: (1) intraday, (2) daily, (3) weekly, and (4) monthly. Daily, weekly, and monthly time series contain up to 20 years of historical data. The intraday time series typically spans the last 10 to 15 trading days.
## Explanations of codes:
There are basically two codes used here one is get_data.py and another one is numerical_algo.py
get_data.py is used to get stock info and store in csv
numerical_algo.py is to get the statistics and analyse the peaks of the stocks
## get_data.py:
#### """All data recieved from this is stored in a zip file in google drive at https://drive.google.com/open?id=1eS2CROcg1sbjUuRLnLSrkSVVGhTqMAT3"""
in get_data.py we have defined some functions called get_stock(), get_all_companies(), parse_intra_day()
get_stock(): collects stock data abut Apple company for every day and store it in csv format and print it to a file. 
get_all_comapnaies(): it will collect details of all companies listed in NASDAQ stock exchange
parse_intra_day(): This function takes chunks of intra day data and converts to csv file for each day stored in the folder days under each stock folder
## numerical_algo.py:
This function is to prepare the statistics of the entire data available.
So there are many parameters considered and calculated to plot the graphs to see the changes in stock price, it will help in model the data set. This function returns the data, the first difference of high price and low price
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
    
## Functions of Google API
This API will pull out all latest news regaridng Apple company from news in Internet. For this we have decided to filter news from many famous news agencies and e-news papers like Financial times, Rueters, CNBC, Wall street journel etc. In order to classify the news in to more useful information to which has influence on the stock market we translate this news in to some numerical values. Well, for this we need to classify the news into positive impact news and negative impact news. 
    
Having said that it is not easy to value some news to be positive or negative, here apply some dictionary words  mentioned in the list https://sraf.nd.edu/textual-analysis/resources/#Master%20Dictionary(Loughran-McDonald Sentiment Word Lists). Usually good news are are constructed with words which have positive meaning or progressive impact. Good news create positive impact for the company and thus for its stocks. Bad news do just the opposite impact on stock market. So how can we classify precisely a news into positive or negative?, well, for this we use some list of frequently used positve words by news agencies in fiancial market. So basically we will check in the overall news how many times these positve and negative words lists are present and each match will count value one with a sign +ve or -ve based on whether it is word with positive impact or negative impact respectievely. The overall values will be sumed up for a particular day. So after analysing some certain news page our algortihm will calculate the value of that particular page and thus predict its impact on next days stock value.
    
    
   
