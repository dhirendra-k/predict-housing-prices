evaluate_project = function(mymain = "sjpatel7_mymain.R"){
  Elasticnet_errors = NULL
  xgboost_errors = NULL
  for (i in 1:10){
    print(paste0("Split ", i))
    source('DataGenerate.R')
    DataGenerate(i)
    test.y <- read.csv("test_y.csv")
    
    source(mymain)
    names(test.y)[2] <- "True_Sale_Price"
    
    # Elasticnet Prediction
    pred1 <- read.csv("mysubmission1.txt")
    pred1 <- merge(pred1, test.y, by="PID")
    RMSE_1 <- sqrt(mean((log(pred1$Sale_Price) - log(pred1$True_Sale_Price))^2))
    
    # XG-boost prediction
    pred2 <- read.csv("mysubmission2.txt")
    pred2 <- merge(pred2, test.y, by="PID")
    RMSE_2 <- sqrt(mean((log(pred2$Sale_Price) - log(pred2$True_Sale_Price))^2))
    
    #print(paste0("RMSE ",i, "(Elasticnet,XG-Boost):    (", RMSE_1,",    ", RMSE_2,")"))
    
    Elasticnet_errors = c(Elasticnet_errors,RMSE_1)
    xgboost_errors = c(xgboost_errors, RMSE_2)
  }
  print('--------------------------------------------------')
  for (i in 1:10) {
    print(paste0("RMSE ",i, "(Elasticnet,XG-Boost):    (", Elasticnet_errors[i],",    ", xgboost_errors[i],")"))
  }
  print(paste0("Mean (Elasticnet,XG-Boost):     (",mean(Elasticnet_errors),",    ",mean(xgboost_errors),")"))
}

evaluate_project()