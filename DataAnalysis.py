#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================
# 
# Using pandas library for data processing
# 
# =============================

#%%
# importing libraries
import time
import requests
import csv
# import get_data
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from scipy import signal
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import pandas_datareader.data as web
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
#%%
import technical_indicators as TI
# neural netowork libraries
#%%
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

# import numerical_algo
#%%
# ====== input parameters =============
seq_len = 22
d = 0.2
shape = [4, seq_len, 1] # feature, window, output
neurons = [128, 128, 32, 1]
epochs = 300
# =====================================
#%%
def load_stock_data_intraday(company,date):
    data_filename = "data/"+company+"/date/"+date+".csv"
    return pd.read_csv(data_filename, parse_dates=['timestamp'])

def load_stock_data_daily(company="AAPL", output_size="compact"):
    data_filename = "data/"+company+"/"+company+"_daily_"+output_size+".csv"
    return pd.read_csv(data_filename, parse_dates=['timestamp'])

def load_stock_data_weekly(company="AAPL"):
    data_filename = company+"/weekly_adjusted_"+company+".csv"
    return pd.read_csv(data_filename, parse_dates=['timestamp'])

def load_news(company="AAPL"):
    return pd.read_csv(company+"/NYT/newsinfo.csv", parse_dates=['date begin'])

def load_file(file="", dateString=""):
    return pd.read_csv(file, parse_dates=[dateString])
#%%
def playground():
    stock_data = load_stock_data_daily("AAPL")
    stock_data.head()
    stock_data.info()
    stock_data.describe()
    stock_data.hist()
    # show the graph
    plt.show()

    stock_data.corr() # see and plot correlations 
    pd.plotting.scatter_matrix(stock_data[["close","volume"]])
    plt.show()
    stock_data.info()

    # [total,diff_hi,diff_low,perc_high,perc_low,f,g,h] =numerical_algo.return_stats_intra("AAPL","2018-04-20",plot=1)

    # print (total)
# ==================================
#%%
def readStockWeb(_stock="AAPL",freq="weekly"):
    start   = datetime.datetime(2000,1,1)
    end     = datetime.datetime.today()
    # read from morningstar
    dat = web.DataReader(_stock,"morningstar",start, end)
    # print(dat)

# function to normalize the stock data
#%%
def normalizeData(df):      
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['adjusted close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    return df
def plot_bargraph(x, _data, _label="no_label", frame="weekly",_color="red"):
    #plot data
    fig, ax = plt.subplots(figsize=(15,7))

    ax.bar(x, _data, color=_color, label=_label)
    plt.legend(loc='best')

    #set ticks per selection
    if(frame == "monthly"):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    elif (frame == "yearly"):
        ax.xaxis.set_major_locator(mdates.YearLocator())
    elif(frame == "weekly"):
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.show()

# create neural network model
#%%
def create_nn_model(layers, neurons, d):
    model = Sequential()
    # [4, 22, 1]
    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))
    # model = load_model('my_LSTM_stock_model1000.h5')
    # adam = keras.optimizers.Adam(decay=0.2)
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
#%%
def training(file, k=5):
    # print (stockData.info())
    stockData = pd.read_csv(file)
    stockData = stockData.dropna()
    # drop the columns open, high and low
    stockData.drop("open", axis=1, inplace=True)
    stockData.drop("high", axis=1, inplace=True)
    stockData.drop("low", axis=1, inplace=True)
    stockData.drop("dividend amount", axis=1, inplace=True)

    # save features to X
    X = stockData.drop("Action", axis=1)
    X = X.drop("timestamp", axis=1)
    # save target to Y - in our case the Action to do (Put/Call)
    Y = stockData["Action"].copy()

    # split into training and test set
    Xtrain, Xtest = train_test_split(X, test_size=0.2, random_state=k)
    Ytrain, Ytest = train_test_split(Y, test_size=0.2, random_state=k)
    # print (train)
    # print(test)
    Xtrain.info()
    Xtest.info()
    
    model = RandomForestClassifier()
    # train the model
    model.fit(Xtrain, Ytrain)
    # test the model
    Yp = model.predict(Xtest)

    # create confusion matrix
    CM = confusion_matrix (Yp, Ytest)
    print (CM)
    accur = metrics.accuracy_score(Ytest,Yp)
    precision = metrics.precision_score(Ytest,Yp, average="macro")
    print ("Accuracy: ",accur,"\nPrecisionScore: ", precision)


    # ===== neural network for prediction ========
    # create neural net
    

    model_nn = create_nn_model(shape, neurons, d)
    model_nn.fit(
        Xtrain,
        Ytrain,
        batch_size=512,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    print(model_nn)



#%%
def process_stock(_stock="AAPL", _print=False, _type="compact",freq="daily" ,_loadnews=False, _normalize=False):
    if (freq == "daily"):
        stockData = load_stock_data_daily(_stock, _type)
    elif (freq == "weekly"):
        stockData = load_stock_data_weekly(company=_stock)

    stockData.set_index('timestamp',inplace=True)

    # drop close column because only adjusted close is interesting because of the stock splits
    stockData.drop('close', axis=1, inplace=True)

    # get first degree difference and create new column
    # Abssolute
    stockData['firstDiffAbs'] = stockData["adjusted close"].diff(periods=-1)
    # percentage
    stockData['firstDiff_%'] = stockData.firstDiffAbs/stockData["adjusted close"] * 100
    
    # add ROC-1
    stockData = TI.rate_of_change(stockData,2)


    # normalize data
    if _normalize:        
        stockData = normalizeData(stockData)

    # plt.plot(stockData['adjusted close'], color='blue')
    # plt.show()

    # save processed file with new columns
    stockData.to_csv(_stock+"/weekly_adjusted_"+_stock+"_processed.csv")
    # stockData.to_csv("AAPL/stockData.csv")
    # stockData.to_csv("AAPL/StockFull.csv")
    
    print ("processed the stock with Symbol: "+_stock)

    return stockData
# =======================================
#%%
def addNews(file="AAPL/weekly_adjusted_AAPL_processed.csv", _loadnews=True,_print=False):
    stockData = pd.read_csv(file, parse_dates=['timestamp'])
    stockData.set_index('timestamp', inplace=True)
    # check if news information shall be loaded for the stock
    if(_loadnews):
        # load apple news nyt and set index
        News = load_file("AAPL/NYT/newsinfo.csv","date begin")
        News.set_index('date begin',inplace=True)

        if(_print): 
            # print news classification
            plot_bargraph(News.index, News['score'],"News Score", "yearly")

        # add news rating as colum to stockdata
        stockData['newsRating'] = News['score']
        # stockData += News.pos_simple

        stockData.info()
        if(_print):
            stockData.corr() # see and plot correlations 
            pd.plotting.scatter_matrix(stockData[["firstDiffAbs","newsRating"]])
            pd.plotting.scatter_matrix(stockData[["adjusted close","newsRating"]])
            plt.show()
    stockData.to_csv("AAPL/weekly_adjusted_AAPL_processed_final.csv")

# data = process_stock(_print=False,_loadnews=True)
# process_stock(freq="weekly",_normalize=False)
# training(file="AAPL/stockData.csv")
# training(file="AAPL/StockFull.csv")
#training(file="AAPL/weekly_adjusted_AAPL_corr.csv")


# newsNYT = pd.read_csv("AAPL/NYT/newsinfo.csv", parse_dates=['date begin'])
# newsNYT.set_index('date begin',inplace=True)
# newsNYT.info()

# addNews()

#%%
file = "C:/Users/user/Documents/GitHub/stock_market_predictor/AAPL/weekly_adjusted_AAPL_corr4.csv"
stockData = pd.read_csv(file)

#%%
stockData = stockData.dropna()
# drop the columns open, high and low
stockData.drop("open", axis=1, inplace=True)
stockData.drop("high", axis=1, inplace=True)
stockData.drop("low", axis=1, inplace=True)
stockData.drop("Action2", axis=1, inplace=True)
stockData.drop("dividend amount", axis=1, inplace=True)
stockData.drop("firstDiffAbs",axis =1, inplace = True)
stockData.drop("firstDiff_%",axis =1, inplace = True)
stockData.drop("ROC_2",axis =1, inplace = True)
stockData.drop("volume",axis =1, inplace = True)


#%%
# save features to X
X = stockData.drop("Action", axis=1)
X = X.drop("timestamp", axis=1)
# save target to Y - in our case the Action to do (Put/Call)
Y = X["adjusted close"].copy()
X = X.drop("adjusted close",axis = 1)
# X = X.iloc[::-1]
# X.sort_index(ascending = True)
# X= X.iloc[::-1]
# split into training and test set
Xtrain, Xtest = train_test_split(X, train_size=0.8, random_state=5623,shuffle = True)
Ytrain, Ytest = train_test_split(Y, train_size=0.8, random_state=5623,shuffle = True)
print(Ytrain)
#print(Xtrain,Ytrain)

#%%
scaler = StandardScaler().fit(Xtrain)
Xtrain_scaled = pd.DataFrame(scaler.transform(Xtrain), index=Xtrain.index.values, columns=Xtrain.columns.values)
Xtest_scaled = pd.DataFrame(scaler.transform(Xtest), index=Xtest.index.values, columns=Xtest.columns.values)

#%%
rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=1254)
rf.fit(Xtrain, Ytrain)

predicted_train = rf.predict(Xtrain)
predicted_test = rf.predict(Xtest)

#%%
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(Xtrain, Ytrain)
# predicted_train = svr_rbf.predict(Xtrain)
predicted_test = svr_rbf.fit(Xtrain, Ytrain).predict(Xtest)


#%%
from sklearn.linear_model import Ridge
ridge = Ridge(fit_intercept=True, alpha=0.5)
ridge.fit(Xtrain,Ytrain)
predicted_test = ridge.predict(Xtest)

#%%

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Xtrain,Ytrain)
predicted_test = ridge.predict(Xtest)

#%%
error = abs(predicted_test - Ytest)
print(error)
errors_per = abs(predicted_test - Ytest)*100/Ytest
# print(errors_per)
print(np.mean(errors_per))
# print((predicted_test))
test_score = r2_score(Ytest, predicted_test)
spearman = spearmanr(Ytest, predicted_test)
pearson = pearsonr(Ytest, predicted_test)

#%%  Cross val
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr, Xtrain, Ytrain, scoring="neg_mean_squared_error", cv=10)
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())


#%%
# print(predicted_test)
# print(Xtest)
    # print(Xtrain.index)
    # plt.plot(Xtrain,Ytrain,'ro')
#plt.plot(Ytest,Xtrain.index'bo')
# plt.plot(Xtrain.index,Ytrain,'go')
plt.plot(Xtest.index,Ytest,marker = '.',ls= '',color='g')
plt.xlabel('time in Weeks')
plt.ylabel('price')
plt.gca().invert_xaxis()
# print(type(Xtest.index))
# print(type(predicted_test))


plt.plot(Xtest.index,predicted_test,marker = '.',ls= '',color='r')
plt.show()
plt.scatter(predicted_test,Ytest)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()

print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
print(f'Test data R-2 score: {test_score:>5.3}')
print(f'Test data Spearman correlation: {spearman[0]:.3}')
print(f'Test data Pearson correlation: {pearson[0]:.3}')
print(mean_squared_error(Ytest,predicted_test))
#%%




####################################################
#Classification
#%%
file = "C:/Users/user/Documents/GitHub/stock_market_predictor/AAPL/weekly_adjusted_AAPL_corr5.csv"
stockData = pd.read_csv(file)

#%%
stockData = stockData.dropna()
# drop the columns open, high and low
stockData.drop("open", axis=1, inplace=True)
stockData.drop("high", axis=1, inplace=True)
stockData.drop("low", axis=1, inplace=True)
# stockData.drop("Action", axis=1, inplace=True)
stockData.drop("dividend amount", axis=1, inplace=True)
stockData.drop("second_diff", axis=1, inplace=True)
stockData.drop("firstDiffAbs",axis =1, inplace = True)
stockData.drop("firstDiff_%",axis =1, inplace = True)
stockData.drop("ROC_2",axis =1, inplace = True)
stockData.drop("volume",axis =1, inplace = True)
stockData.drop("n-1",axis =1, inplace = True)
stockData.drop("n-2",axis =1, inplace = True)
stockData.drop("n-3",axis =1, inplace = True)
stockData.drop("n-4",axis =1, inplace = True)
stockData.drop("n-5",axis =1, inplace = True)
stockData.drop("n-6",axis =1, inplace = True)
stockData.drop("n-7",axis =1, inplace = True)

stockData.drop("n-8",axis =1, inplace = True)

#%%
# save features to X
X = stockData.drop("Action2", axis=1)
X = X.drop("timestamp", axis=1)
# save target to Y - in our case the Action to do (Put/Call)
Y = X["Action"].copy()
X = X.drop("Action", axis=1)

# X = X.drop("adjusted close",axis = 1)
# split into training and test set
Xtrain, Xtest = train_test_split(X, test_size=0.2, random_state=5623,shuffle = True)
Ytrain, Ytest = train_test_split(Y, test_size=0.2, random_state=5623,shuffle = True)
print(Ytrain)


#%%

rfc = RandomForestClassifier(n_jobs=2, random_state=345)
rfc.fit(Xtrain,Ytrain)
predicted_test = rfc.predict(Xtest)

#%%
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(Xtest, Ytest)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(Xtest.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

print(Xtest)



#%%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(Xtrain,Ytrain)
predicted_test = knn.predict(Xtest)
#%%



print(predicted_t   est)
#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
confusion_matrix(Ytest,predicted_test)
accuracy_score(Ytest,predicted_test)
# precision_score(Ytest,predicted_test)
