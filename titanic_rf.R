library(rpart)# tree models 
library(caret) # feature selection
library(rpart.plot) # plot dtree
library(ROCR) # model evaluation
library(e1071) # tuning model
library(RColorBrewer)
library(rattle)# optional, if you can't install it, it's okay
library(tree)
library(ISLR)
library(randomForest)
library(rpart.plot)
library(dplyr)


setwd("C:\\Users\\ADMIN\\Desktop\\R Models\\Decision Tree")
Carseats <- read.csv("Titanic.csv")
head(Carseats)
tail(Carseats)
str(Carseats)
summary(Carseats)

Carseats <- Carseats[ -c(1,4,9,11) ]
## Let's also change the labels under the "status" from (0,1) to (normal, abnormal)   
Carseats$Pclass <- as.factor(Carseats$Pclass) 
Carseats$Survived <- factor(Carseats$Survived, levels = c(0, 1),labels = c('No', 'Yes'))

## Check the missing value (if any)
sapply(Carseats, function(x) sum(is.na(x)))

Carseats <- na.omit(Carseats)

## Now you can randomly split your data in to 70% training set and 30% test set   
set.seed(123)
train <- sample(1:nrow(Carseats), round(0.70*nrow(Carseats),0))
test <- -train
training <- Carseats[train,]
testing <- Carseats[test,]
test_Survived <- testing$Survived

# Create a Random Forest model with default parameters
fit <- randomForest(Survived~., data = training, ntree = 500, mtry = 2, maxnodes = NULL, importance = TRUE)

# Define the control
train()
trControl <- trainControl(method = "cv",
    number = 10,
    search = "grid")
# Run the model
rf_default <- train(Survived~., data = training, method = "rf", metric = "Accuracy", trControl = trControl)
# Print the results
print(rf_default)

#testing the model with mtry 1 to 10
set.seed(1234)
tuneGrid <- expand.grid(.mtry = c(1: 10))
rf_mtry <- train(Survived~., data = training, method = "rf",
             metric = "Accuracy", tuneGrid = tuneGrid, trControl = trControl,
                 importance = TRUE, nodesize = 14, ntree = 300)
print(rf_mtry)
rf_mtry$bestTune$mtry
best_mtry <- rf_mtry$bestTune$mtry 
best_mtry
max(rf_mtry$results$Accuracy)

#Search the best maxnodes
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(3: 25)) {
    set.seed(1234)
    rf_maxnode <- train(Survived~.,
        data = training,
        method = "rf",
        metric = "Accuracy",
        tuneGrid = tuneGrid,
        trControl = trControl,
        importance = TRUE,
        nodesize = 14,
        maxnodes = maxnodes,
        ntree = 300)
    current_iteration <- toString(maxnodes)
    store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(17: 37)) {
    set.seed(1234)
    rf_maxnode <- train(Survived~.,
        data = training,
        method = "rf",
        metric = "Accuracy",
        tuneGrid = tuneGrid,
        trControl = trControl,
        importance = TRUE,
        nodesize = 14,
        maxnodes = maxnodes,
        ntree = 300)
    current_iteration <- toString(maxnodes)
    store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

#Search the best ntrees
store_maxtrees <- list()
for (ntree in c(200,250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
    set.seed(5678)
    rf_maxtrees <- train(Survived~.,
        data = training,
        method = "rf",
        metric = "Accuracy",
        tuneGrid = tuneGrid,
        trControl = trControl,
        importance = TRUE,
        nodesize = 14,
        maxnodes = 24,
        ntree = ntree)
    key <- toString(ntree)
    store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)

#final model
fit_rf <- train(Survived~.,
    training,
    method = "rf",
    metric = "Accuracy",
    tuneGrid = tuneGrid,
    trControl = trControl,
    importance = TRUE,
    nodesize = 14,
    ntree = 250,
    maxnodes = 19)

#predicting randomforest
predictions <- predict(fit_rf, testing)
mean(predictions != test_Survived)


conf.matrix <- table(testing$Survived, predictions)
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ":")
colnames(conf.matrix) <- paste("Predicted", colnames(conf.matrix), sep = ":")
print(conf.matrix)

varImpPlot(fit_rf)



table_mat <- table(testing$Survived, predictions)
table_mat
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))

round(sum(diag(table_mat)) / sum(table_mat),2)