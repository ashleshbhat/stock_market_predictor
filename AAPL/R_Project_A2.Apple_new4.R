library(caret)
library(dplyr)
library(lubridate)
library(class)

stocks <- read.csv(file = "/Users/Pahel009/Desktop/AAPL/stockData_1.csv",sep = ',',header = TRUE)

dim(stocks)
summary(stocks)

#data Slicing
set.seed(100)
indxTrain <- createDataPartition(y = stocks$Action, p = 0.75, list = FALSE)
training <- stocks[indxTrain,]
testing <- stocks[-indxTrain,]

prop.table(table(training$Action)) * 100



trainX <- training[,names(training) != "Action"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues

set.seed(200)

### Knn Model
ctrl <- trainControl(method="repeatedcv",repeats = 6) 
knnFit <- train(Action ~ ., data = training, method = "knn", trControl = ctrl, 
                preProcess = c("center","scale"), tuneLength = 20)

plot(knnFit)

#Output of kNN fit
knnFit

knnPredict <- predict(knnFit,newdata = testing )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, testing$Action )
