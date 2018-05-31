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

#%%
file = "C:/Users/user/Documents/GitHub/stock_market_predictor/AAPL/weekly_adjusted_AAPL_corr5.csv"
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
# stockData.drop("diff%-1",axis =1, inplace = True)
stockData.drop("diff%-2",axis =1, inplace = True)
# stockData.drop("second_diff-1",axis =1, inplace = True)
stockData.drop("second_diff",axis =1, inplace = True)
stockData = stockData.drop("Action", axis=1)
# stockData.drop("n-2",axis =1, inplace = True)
# stockData.drop("n-3",axis =1, inplace = True)
# stockData.drop("n-4",axis =1, inplace = True)
# stockData.drop("n-5",axis =1, inplace = True)
# stockData.drop("n-6",axis =1, inplace = True)
# stockData.drop("n-7",axis =1, inplace = True)
# stockData.drop("n-8",axis =1, inplace = True)
# stockData.drop("n-10",axis =1, inplace = True)
# print(stockData)




#%%
# save features to X
X = stockData.drop("timestamp", axis=1)
# save target to Y - in our case the Action to do (Put/Call)
Y = X["adjusted close"].copy()
X = X.drop("adjusted close",axis = 1)
# X = X.iloc[::-1]
# X.sort_index(ascending = True)
# X= X.iloc[::-1]
# split into training and test set
print(X)
Xtrain, Xtest = train_test_split(X, train_size=0.8, random_state=5623,shuffle = True)
Ytrain, Ytest = train_test_split(Y, train_size=0.8, random_state=5623,shuffle = True)
# print(Ytrain)
print(Xtrain,Ytrain)

# #%%
# scaler = StandardScaler().fit(Xtrain)
# Xtrain_scaled = pd.DataFrame(scaler.transform(Xtrain), index=Xtrain.index.values, columns=Xtrain.columns.values)
# Xtest_scaled = pd.DataFrame(scaler.transform(Xtest), index=Xtest.index.values, columns=Xtest.columns.values)

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
scores = cross_val_score(rf, Xtrain, Ytrain, scoring="neg_mean_squared_error", cv=3)
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())


#%%

import matplotlib.patches as mpatches
plt.plot(Xtest.index,Ytest,marker = '.',ls= '',color='g',markersize=3)
plt.xlabel('time in Weeks')
plt.ylabel('price')
plt.gca().invert_xaxis()

plt.plot(Xtest.index,predicted_test,marker = '.',ls= '',color='r',markersize=3)
red_patch = mpatches.Patch(color='red', label='Predicted')
green_patch = mpatches.Patch(color='green', label='Actual')

plt.legend(handles=[red_patch, green_patch])
# plt.savefig('svm-1_1.png')
# plt.savefig('ridge-1_1.png')
# plt.savefig('random_forest-1_1.png')
plt.savefig('linear_regression-1_1.png')
plt.show()

plt.scatter(predicted_test,Ytest)
plt.xlabel('predicted')
plt.ylabel('actual')
# plt.savefig('ridge-2_1.png')
# plt.savefig('random_forest-2_1.png')
# plt.savefig('svm-2_1.png')    
plt.savefig('linear_regression-2_1.png')
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



print(predicted_test)
#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
confusion_matrix(Ytest,predicted_test)
accuracy_score(Ytest,predicted_test)
# precision_score(Ytest,predicted_test)
