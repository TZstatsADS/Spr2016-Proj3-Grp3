install.packages('data.table')
install.packages('e1071')
library(data.table)
library(e1071)
####################################################

train <- function(dat_train, label_train){
  train.x <- dat_train
  train.y <- as.factor(label_train)
  trainadv.x <- train.x[, 1:9735]
  trainbase.x <- train.x[, 9736:dim(train.x)[2]]
  SVM_Model_adv <- svm(trainadv.x, train.y, cost = 100)
  SVM_Model_base <- svm(trainbase.x, train.y)
  #return(fit_train=SVM_Model_adv)
  return(fit_train=list(SVM_Model_adv=SVM_Model_adv, SVM_Model_base=SVM_Model_base))
}

