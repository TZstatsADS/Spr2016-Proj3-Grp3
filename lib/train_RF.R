install.packages('randomForest')
install.packages('e1071')
library(randomForest)
library(e1071)
train.data <- read.csv('train_data.csv', header = TRUE)
train.data.x <- train.data[, 4:515]
train.data.y <- as.factor(train.data$X0)
model1 <- randomForest(train.data.x, train.data.y, ntree = 1000)
feature_importance <- as.data.frame(importance(model))
feature_name <- paste(rep('x', 512), 1:512, sep = '')
feature_importance <- cbind(feature_importance, feature_name)
feature_importance <- feature_importance[order(-feature_importance$MeanDecreaseGini),]
select_feature <- feature_importance[feature_importance$MeanDecreaseGini > 1.8,]$feature_name
select_train.data.x <- train.data.x[, select_feature]
model2 <- randomForest(select_train.data.x, train.data.y, ntree = 1000)

cost_list <- c(0.01, 0.1, 1, 2.7, 10, 100, 150, 200, 250, 300, 350)
gamma_list <- c(0.0001, 0.0005, 0.0007, 0,001, 0.01, 0.09, 0.015, 0.02, 0.025, 0.03, 0.1, 1)
for(i in cost_list) {
    for(j in gamma_list){
        set.seed(10)
        model <- svm(train.data.x, train.data.y, kernel = 'radial',gamma = j, cost = i, cross = 10)
        accuracy <- summary(model)$tot.accuracy
        observation <- c(i, j, accuracy)
        print(observation)
    }
}