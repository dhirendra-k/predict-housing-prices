**Predict the Housing Prices in Ames**

1. **Introduction:**
 The Ames dataset was analyzed with the goal of forming two prediction models for housing prices. We chose to construct an Elasticnet linear regression model and a gradient boosting tree model. Elasticnet predictions are output in &quot;mysubmission1.txt&quot;, and the boosting tree model predictions are output in &quot;mysubmission2.txt&quot;.

For all results in this report, we used a seed of &quot;set.seed(6017)&quot; in our code file for reproducibility.

1. **Pre-processing**

  1. **Imputation**

While inspecting the Ames dataset for missing values, the only variable containing &quot;NA&quot; values was the discrete numerical variable &quot;Garage\_Yr\_Blt&quot;. The &quot;NA&quot; value for &quot;Garage\_Yr\_Blt&quot; exactly corresponds to the &quot;No\_Garage&quot; level for the categorical variable &quot;Garage\_Finish&quot;. Due to this exact correspondence in values, we replaced the &quot;NA&quot; values for &quot;Garage\_Yr\_Blt&quot; with the value &quot;0&quot;.

This variable also contains an incorrect value &quot;2207&quot;. We assumed this was an input error for the value &quot;2007&quot; and we replaced this value with 2007 if it occurred in the test or train data.

  1. **Winsorization**

Winsorization was applied on all continuous numeric variables included in our model. We imposed an upper limit of the 95% quantile value to the continuous numerical variables.

  1. **Categorical to Dummy Variable Conversion**

For the gradient boosting tree model, categorical variables with _k_ levels were converted into _k_ binary numerical variables for _k_ \&gt; 2. If the categorical variable has only 2 levels, it is converted into a single binary numerical variable. This preprocessing step is necessary for tree model training, as tree models do not learn intercepts and hence require a binary variable for each level (if the number of levels is above 2). XGBoost requires numerical data, and any factors must be converted into numeric values.

  1. **Variable Removal**

The following variables were removed from the training and test data.

| **Removed Variable** | **Reason** |
| --- | --- |
|
| |
| PID | Not an explanatory variable |
| Street | Contains type of road access and it&#39;s not useful to use for price prediction |
| Utilities | Most of the data having AllPub (All public Utilities) that is why this variable wont much impact in price analysis |
| Condition\_2 | Nearly all values are &quot;norm&quot; for Normal. |
| Roof\_Mat1 | Most of the data having &quot;CompShg&quot; Standard (Composite) Shingle |
| Heating | Nearly all values are &quot;GasA&quot;. Also, this variable is better explained by &quot;HeatingQC&quot; |
| Pool\_QC | There was not enough categorical data to use this variable |
| Misc\_Feature | Very little data for this feature. Most values are &quot;None&quot;. Potential for overfitting |
| Misc\_Value | This Variable corresponds to Misc\_Feature. |
| Low\_Qual\_Fin\_SF |
 |
| Pool\_Area |
 |
| Longitude | Neighborhood is a more relevant feature for location. There is unlikely a linear relationship between geological coordinates and housing price. This may cause overfitting for Lasso. |
| Latitude | See &quot;Longitude&quot; reason. |

1. **Elasticnet**

We used the _glmnet_ library (specifically the function _cv.glmnet()_) to train our Elasticnet model. The ƛ corresponding to the minimum CV error from _cv.glmnet()_ was used to predict the test data. The Elasticnet mixing parameter 𝞪 was tuned in 0.1 increments. The figure below shows the mean RMSE for 10 train/test splits of the Ames Housing Dataset. An 𝞪 value of 0.1 minimized our test error, and we used this value in our model. We found that Elasticnet performed better than pure Ridge and pure Lasso models for our preprocessed data. Test error may be improved by filtering more features based on variable importance.

![](RackMultipart20210224-4-d5ti0_html_c145482586474db1.png)

1. **Gradient Boosting Tree**

We used the library _xgboost_ to construct our gradient boosting tree model.

  1. **Tuning Parameters:**

The parameters for xgboost were initialized as the default parameters

- _xgboost(eta=0.3, max\_depth=6, gamma=0,subsample=1,colsample\_bytree=1, ...)_) and _nrounds_ set to 100.

Using the function _xgb.cv()_, we tested adjustments for the parameters using 5-fold CV error on the train dataset. The best performing tuning parameters that we tested on our training sets were

- _eta=0.05, max\_depth=3, gamma=0, subsample=0.9, colsample\_bytree=1, nrounds=1000_

Interestingly. The gamma and colsample\_bytree parameters performed better at their default values in conjunction with the other tuned parameters. This may be due to the low eta value and max tree depth reducing the likelihood of overfitting this training dataset.

**Accuracy**

|
 | **Elasticnet Test RMSE** | **Gradient Boosting Tree**** Test RMSE **|** Elasticnet Training Time (seconds) **|** Gradient Boosting Tree Training Time (seconds)** |
| --- | --- | --- | --- | --- |
| **Test Split 1** | 0.1288887794124 | 0.1134992190452 | 1.50623 | 6.11706 |
| **Test Split 2** | 0.1200980787998 | 0.118131181898 | 1.89341 | 6.58287 |
| **Test Split 3** | 0.1406432340009 | 0.1093656366518 | 1.42131 | 6.09144 |
| **Test Split 4** | 0.1229512291741 | 0.1140083969747 | 1.63419 | 5.92381 |
| **Test Split 5** | 0.116416400440 | 0.1124747477200 | 1.54611 | 6.22273 |
| **Test Split 6** | 0.1325913458355 | 0.1268891984706 | 1.40511 | 6.08655 |
| **Test Split 7** | 0.1289203505460 | 0.1319895942973 | 1.71696 | 6.96721 |
| **Test Split 8** | 0.1191563561479 | 0.122351832374 | 1.38167 | 5.80701 |
| **Test Split 9** | 0.1338990395857 | 0.1271712660921 | 1.76232 | 6.39109 |
| **Test Split 10** | 0.1232845342836 | 0.1237556133332 | 1.47449 | 6.1247 |
|
| |
| **Mean** | **0.126684934822** | **0.119963668685** | **1.57418** | **6.23145** |

The Gradient boosting tree model performed much better than the Elasticnet regression model. However, training speed of the boosting tree model, especially with the low eta value and high boosting rounds chosen, was much slower than the Elasticnet training speed. Elasticnet may benefit from more in depth variable selection processes.

**Testing Specs**

1. Windows 10, Intel i5-7300HQ CPU @ 2.50 GHz, 16GB RAM