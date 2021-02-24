# 
# COURSE: CS 598: Practical Statistical Learning ( Fall 2020 ) 
# Project: Project-1
# Details: Project based on prediction models:
#             1 - Elasticnet
#             2 - BOOSTING TREE
#
# Team members - (2)
#           - NetID : sjpatel7
#             Name : Shawn J Patel
#
#           - NetID : dk21
#             Name : Dhirendra Kumar
#


# Include libraries
if (!require("pacman"))
  install.packages("pacman")
pacman::p_load("glmnet",
               "xgboost",
               "mgcv")

# Set seed for reproducibility
set.seed(6017)



## ******* User defined method to preprocess training data *******
Preprocess_Elasticnet = function(data) {
  #Impute NA values in "Garage_Yr_Blt" with 0
  data$Garage_Yr_Blt[is.na(data$Garage_Yr_Blt)] = 0
  #Assume Garage_Yr_Blt "2207" value corresponds to 2007
  data$Garage_Yr_Blt[data$Garage_Yr_Blt == 2207] <- 2007
  
  #Remove Variables
  remove.vars <-
    c(
      'Utilities',
      'Electrical',
      'Fireplaces',
      'Condition_2',
      'Heating',
      'Pool_QC',
      'Misc_Feature',
      'Misc_Value',
      'Pool_Area',
      'Longitude',
      'Latitude',
      "PID"
    )
  data = data[,!colnames(data) %in% remove.vars]
  
  #Winsorization
  winsor.vars <-
    c(
      "Lot_Frontage",
      "Lot_Area",
      "Mas_Vnr_Area",
      "BsmtFin_SF_2",
      "Bsmt_Unf_SF",
      "Total_Bsmt_SF",
      "Second_Flr_SF",
      'First_Flr_SF',
      "Gr_Liv_Area",
      "Garage_Area",
      "Wood_Deck_SF",
      "Open_Porch_SF",
      "Screen_Porch"
    )
  for (var in winsor.vars) {
    # apply winsorization on every numerical variables
    p <- 0.95
    thresh <- quantile(data[, var], p)
    rp.index <- which(data[, var] > thresh)
    data[rp.index, var] <- thresh
  }
  
  #Transform Categorical Variables into dummy variables
  #Followed instructor Feng Liang's implementation for this process
  #See Piazza post @261
  categorical.vars <- colnames(data)[which(sapply(data, function(x) is.factor(x)))]
  data.tmp <- data[, !colnames(data) %in% categorical.vars, drop=FALSE]
  
  #build dummy variable matrix and column list
  for (var in categorical.vars){
    #add matrix for this variable to the categoricalLess matrix
    levels <- sort(unique(data[, var]))
    m <- length(levels)
    m <- ifelse(m > 2, m, 1)
    data.cat <- matrix(0, nrow(data.tmp), m)
    col.names <- NULL
    for (j in 1:m) {
      data.cat[data[, var] == levels[j], j] <- 1
      col.names <- c(col.names, paste(var, '_', levels[j], sep=''))
    }
    colnames(data.cat) <- col.names
    data.tmp <- cbind(data.tmp, data.cat)
  }
  data <- data.tmp
  
  return(data)
}

Preprocess_XGBoost = function(data) {
  #Impute NA values in "Garage_Yr_Blt" with 0
  data$Garage_Yr_Blt[is.na(data$Garage_Yr_Blt)] = 0
  #Assume Garage_Yr_Blt "2207" value corresponds to 2007
  data$Garage_Yr_Blt[data$Garage_Yr_Blt == 2207] <- 2007
  #Remove variables
  remove.vars <-
    c(
      'Utilities',
      'Electrical',
      'Fireplaces',
      'Heating',
      'Pool_QC',
      'Misc_Feature',
      'Condition_2',
      'Misc_Value',
      'Pool_Area',
      'Longitude',
      'Latitude',
      "PID"
    )
  data = data[,!colnames(data) %in% remove.vars]
  
  #Winsorization
  winsor.vars <-
    c(
      "Lot_Frontage",
      "Lot_Area",
      "Mas_Vnr_Area",
      "BsmtFin_SF_2",
      "Bsmt_Unf_SF",
      "Total_Bsmt_SF",
      "Second_Flr_SF",
      'First_Flr_SF',
      "Gr_Liv_Area",
      "Garage_Area",
      "Wood_Deck_SF",
      "Open_Porch_SF",
      "Screen_Porch"
    )
  for (var in winsor.vars) {
    # apply winsorization on every numerical variables
    p <- 0.95
    thresh <- quantile(data[, var], p)
    rp.index <- which(data[, var] > thresh)
    data[rp.index, var] <- thresh
  }
  
  #Transform Categorical Variables into dummy variables for tree model
  #Followed instructor Feng Liang's implementation for this process
  #See Piazza post @261
  categorical.vars <- colnames(data)[which(sapply(data, function(x) is.factor(x)))]
  data.tmp <- data[, !colnames(data) %in% categorical.vars, drop=FALSE]
  
  #build dummy variable matrix and column list
  for (var in categorical.vars){
    #add matrix for this variable to the categoricalLess matrix
    levels <- sort(unique(data[, var]))
    m <- length(levels)
    m <- ifelse(m > 2, m, 1)
    data.cat <- matrix(0, nrow(data.tmp), m)
    col.names <- NULL
    for (j in 1:m) {
      data.cat[data[, var] == levels[j], j] <- 1
      col.names <- c(col.names, paste(var, '_', levels[j], sep=''))
    }
    colnames(data.cat) <- col.names
    data.tmp <- cbind(data.tmp, data.cat)
  }
  data <- data.tmp
  
  return(data)
}


# Read training and test data
train <- read.csv("train.csv")
test <- read.csv("test.csv")


## ********** (1) - PREDICTION MODEL => "Elasticnet" **********

train.Elasticnet <- Preprocess_Elasticnet(train)
test.Elasticnet <- Preprocess_Elasticnet(test)
train.Elasticnet.y <- log(train.Elasticnet$Sale_Price)

##########################

train.Elasticnet <- model.matrix( ~ ., train.Elasticnet)[, -1]
test.Elasticnet <- model.matrix( ~ ., test.Elasticnet)[, -1]

drop.trainindex <-
  setdiff(unlist(dimnames(train.Elasticnet)[2]), unlist(dimnames(test.Elasticnet)[2]))
train.Elasticnet <-
  train.Elasticnet[, !colnames(train.Elasticnet) %in% drop.trainindex]
drop.testindex <-
  setdiff(unlist(dimnames(test.Elasticnet)[2]), unlist(dimnames(train.Elasticnet)[2]))
test.Elasticnet <-
  test.Elasticnet[, !colnames(test.Elasticnet) %in% drop.testindex]

#####################
t1 = Sys.time()
# Perform cross validation on Elasticnet
cv.out <- cv.glmnet(train.Elasticnet, train.Elasticnet.y, alpha = 0.1)
t2 = Sys.time()

# Perform prediction on Elasticnet with lambda.min
tmp.Elasticnet <-
  predict(cv.out, s = cv.out$lambda.min, newx = test.Elasticnet)

# Write Elasticnet prediction results to file mysubmission1.txt
write.table(
  cbind(test[, 'PID'] , exp(tmp.Elasticnet)),
  file = "mysubmission1.txt",
  sep = ", ",
  row.names = FALSE,
  col.names = c('PID', 'Sale_Price'),
  quote = FALSE
)


## ********** (2) - PREDICTION MODEL => "BOOSTING TREE" *********

train.xgb <- Preprocess_XGBoost(train)
test.xgb <- Preprocess_XGBoost(test)
train.xgb.y <- log(train.xgb$Sale_Price)

##########################

train.xgb <- model.matrix( ~ ., train.xgb)[, -1]
test.xgb <- model.matrix( ~ ., test.xgb)[, -1]

drop.trainindex <-
  setdiff(unlist(dimnames(train.xgb)[2]), unlist(dimnames(test.xgb)[2]))
train.xgb <-
  train.xgb[, !colnames(train.xgb) %in% drop.trainindex]
drop.testindex <-
  setdiff(unlist(dimnames(test.xgb)[2]), unlist(dimnames(train.xgb)[2]))
test.xgb <-
  test.xgb[, !colnames(test.xgb) %in% drop.testindex]

trainD.xgb <- xgb.DMatrix(data=train.xgb, label=train.xgb.y)

#####################
t3 = Sys.time()

#set.seed(6017)
params.tuned <- list(booster='gbtree', eta=0.05, max_depth=3, gamma=0, subsample=0.9, colsample_bytree=1)
#test params.tuned with xgb.cv
#xgbcv <- xgb.cv(params = params.tuned, data = trainD.xgb, nrounds = 1000, nfold = 5, showsd=T, stratified = T, print_every_n = 50, early_stopping_rounds = 40)

xgb = xgboost(
  data = train.xgb,
  label = train.xgb.y,
  params = params.tuned,
  nrounds = 1000,
  verbose = FALSE
)
t4 = Sys.time()

xgb.predict = exp(predict(xgb, as.matrix(test.xgb)))

output = cbind(test$PID, xgb.predict)
colnames(output) = c("PID", "Sale_Price")
write.csv(output, "mysubmission2.txt", row.names = FALSE)
