test <- function(fit_train, dat_test){
  testadv.x <- dat_test[, 1:9735]
  testbase.x <- dat_test[, 9736:dim(dat_test)[2]]
  SVM_Model_adv <- fit_train$SVM_Model_adv
  SVM_Model_base <- fit_train$SVM_Model_base
  adv <- predict(SVM_Model_adv, testadv.x)
  baseline <- predict(SVM_Model_base, testbase.x)
  #return(pred=adv)
  return(pred=list(adv=adv, baseline=baseline))
}