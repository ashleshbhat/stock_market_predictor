# Plots of Results
## Final Result Comparison of Mean Squared Error
| Regression Algorithm        | Input parameters 1           | Input parameters 2|
| -------------               |:-------------:              | -----:           |
| Random Forest      | 4.63 | 4.99 |
| Linear Regression      | 4.766     | 4.73   |
| Ridge Regression       | 4.766 | 4.73|
| Support Vector Machine |   866.04    | 1524.60    |

## Input Parameter
We tried regression with two different input parameters the first one being a basic and second one with more
### Input parameters 1
Some of the Basic Parameters used

|parameter name |detail|
|--- | --- |
|`n-1`|      price 1 week ago|
|`m_avg_5`  |average price of last 5 weeks|
|`max_10`   |maximum price of last 10 weeks|
|`min_10`   |minimum price of last 10 weeks|
|`score`    |advance sentiment analysis score|
|`pos_simple`  |basic sentiment analysis positive score |
|`neg_simple`  |basic sentiment analysis negative score|


### Input parameters 2
Extra parameters used over the previous

|parameter name |detail|
|--- | --- |
|`n-2`|price of 2 weeks ago|
|`n-3`|price of 3 weeks ago|
|`n-4`|price of 4 weeks ago|
|`n-5`|price of 5 weeks ago|
|`diff%-1| percentage difference of price between 1 week and 2 weeks ago| 



# Individual Results
## Random Forest Regression
### Input parameters 1
![Regression Result](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/random_forest-1_1.png)
![scatter plot](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/random_forest-2_1.png)

### Input parameters 2
![Regression Result](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/random_forest-1_2.png)
![scatter plot](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/random_forest-2_2.png)

## Linear Regression and Ridge
### Input parameters 1
![Regression Result](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/linear_regression-1_1.png)
![scatter plot](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/linear_regression-2_1.png)

### Input parameters 2
![Regression Result](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/linear_regression-1_2.png)
![scatter plot](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/linear_regression-2_2.png)

## Support Vector Machines
### Input parameters 1
![Regression Result](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/svm-1_1.png)
![scatter plot](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/svm-2_1.png)

### Input parameters 2
![Regression Result](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/svm-1_2.png)
![scatter plot](https://github.com/ashleshbhat/stock_market_predictor/blob/master/Plots%20for%20Wiki/svm-2_2.png)



