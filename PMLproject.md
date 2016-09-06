# ##Activity Quality Prediction using Weight Lifting Exercise Dataset

By: ZG

### _Overview_
The Weight Lifting Exercise dataset contains experiment data collected from wearable fitness devices of 6 participants who were asked to perform weight lifting exercises.  The experiments were focused on the Quality of the exercises.  More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

In this project, we used the dataset to build a prediction model.  The goal of the project is to predict the manner in which the exercise was performed. 

### _DATA Description_
The data for this project was obtained from this source: http://groupware.les.inf.puc-rio.br/har.  

The training data was provided here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data was provided here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The prediction outcome (the manner in which the exercise was performed) is represented by the "classe" variable in the training set. There are five possible outcomes:

- Exactly according to the specification (Class A), 
- Throwing the elbows to the front (Class B), 
- Lifting the dumbbell only halfway (Class C), 
- Lowering the dumbbell only halfway (Class D) and 
- Throwing the hips to the front (Class E)

### _Data Processing_
In this section we load and explore the data set.


```r
## load the required packages
library(caret)
library(randomForest)
library(gbm)

## load data sets from the working directory into R
training <- read.csv("~/pml-training.csv")
testing <- read.csv("~/pml-testing.csv")

## explore the data
dim(training)
```

```
## [1] 19622   160
```

```r
str(training$classe)
```

```
##  Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```


#### _Feature Selection_
There are 160 variables in the dataset, however a large proportion of them are nulls. We'll exclude them so we'll have a smaller, valid subset of variables that can be used as predictors.  We'll also exclude the userid and timestamp related columns.

Here we pick the relevant subset of the variables (measurements excluding the null columns). 

```r
varnames <- names(training)
varIndx <- c(grep("^roll", varnames),grep("^pitch", varnames), grep("^yaw", varnames),
           grep("^total", varnames),grep("^gyros", varnames), 
           grep("^accel", varnames), grep("^magnet", varnames))
```

The following variables will be used as predictors

```r
features <- varnames[varIndx]
print(features)
```

```
##  [1] "roll_belt"            "roll_arm"             "roll_dumbbell"       
##  [4] "roll_forearm"         "pitch_belt"           "pitch_arm"           
##  [7] "pitch_dumbbell"       "pitch_forearm"        "yaw_belt"            
## [10] "yaw_arm"              "yaw_dumbbell"         "yaw_forearm"         
## [13] "total_accel_belt"     "total_accel_arm"      "total_accel_dumbbell"
## [16] "total_accel_forearm"  "gyros_belt_x"         "gyros_belt_y"        
## [19] "gyros_belt_z"         "gyros_arm_x"          "gyros_arm_y"         
## [22] "gyros_arm_z"          "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [25] "gyros_dumbbell_z"     "gyros_forearm_x"      "gyros_forearm_y"     
## [28] "gyros_forearm_z"      "accel_belt_x"         "accel_belt_y"        
## [31] "accel_belt_z"         "accel_arm_x"          "accel_arm_y"         
## [34] "accel_arm_z"          "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "accel_forearm_x"      "accel_forearm_y"     
## [40] "accel_forearm_z"      "magnet_belt_x"        "magnet_belt_y"       
## [43] "magnet_belt_z"        "magnet_arm_x"         "magnet_arm_y"        
## [46] "magnet_arm_z"         "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [49] "magnet_dumbbell_z"    "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"
```

### _Cross Validation_
The testing dataset, pml-testing.csv,  will be left untouched so that it will be used only once in the end to validate the model.

Cross validation will be performed by splitting the training dataset, pml-training.csv, into a training and testing set.  

I used K-fold cross validation with K=3.  Because the dataset is large (19,622 observations), a 3-fold cross validation should be fairly accurate as there will be plenty of examples to train on.  


```r
smalltrain <- training[,c(features, 'classe')] ##subset the chosen features

set.seed(345)
trainIndx <- createDataPartition(smalltrain$classe, p=0.75, list=F)
trainset <- smalltrain[trainIndx,]
testset <- smalltrain[-trainIndx,]

trCtrl <- trainControl(method='cv', number = 3)
```

### _Prediction Model Selection_
In this section we build two prediction models; one using Random Forest and one with Gradient Boosting algorithm, and evaluate the accuracy of the predictions from the two models.  

#### _1. Prediction using Random Forest_

```r
## fit a RF model
set.seed(123)
modRF <- train(classe ~.,data=trainset,method="rf", trControl = trCtrl) 

##predict classe on the testset using the RF model
set.seed(456)
predRF <- predict(modRF, testset)

## evaluate prediction
confusionMatrix(predRF, testset$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    2    0    0    0
##          B    0  946    5    0    0
##          C    0    1  846   12    0
##          D    0    0    4  792    2
##          E    1    0    0    0  899
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9945         
##                  95% CI : (0.992, 0.9964)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.993          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9968   0.9895   0.9851   0.9978
## Specificity            0.9994   0.9987   0.9968   0.9985   0.9998
## Pos Pred Value         0.9986   0.9947   0.9849   0.9925   0.9989
## Neg Pred Value         0.9997   0.9992   0.9978   0.9971   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1929   0.1725   0.1615   0.1833
## Detection Prevalence   0.2847   0.1939   0.1752   0.1627   0.1835
## Balanced Accuracy      0.9994   0.9978   0.9931   0.9918   0.9988
```

#### _2. Prediction using GBM_

```r
## fit a GMB model
set.seed(123)
modGBM <- train(classe ~.,data=trainset,method="gbm", trControl = trCtrl, verbose=FALSE)

##predict classe on the testset using the GBM model
set.seed(456)
predGBM <- predict(modGBM, testset)

## evaluate prediction
confusionMatrix(predGBM, testset$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1373   43    0    0    1
##          B   14  872   21    2   13
##          C    4   29  820   27    7
##          D    3    3   11  768   17
##          E    1    2    3    7  863
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9576          
##                  95% CI : (0.9516, 0.9631)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9463          
##  Mcnemar's Test P-Value : 3.569e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9842   0.9189   0.9591   0.9552   0.9578
## Specificity            0.9875   0.9874   0.9835   0.9917   0.9968
## Pos Pred Value         0.9689   0.9458   0.9245   0.9576   0.9852
## Neg Pred Value         0.9937   0.9807   0.9913   0.9912   0.9906
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2800   0.1778   0.1672   0.1566   0.1760
## Detection Prevalence   0.2889   0.1880   0.1809   0.1635   0.1786
## Balanced Accuracy      0.9858   0.9531   0.9713   0.9735   0.9773
```

### _Conclusion/Results_

We can see from the Confusion Matrix output of both models above that the Random Forest model performed slightly better than the GBM model.  The Random Forest model had accuracy of 0.9945, and the GBM model had accuracy of 0.9576.  

We'll choose the Random Forest model and use it to predict on the final validation set.

### _Expected out-of-sample Error_
With the Random Forest classifier, we achieved a prediction accuracy of 0.9945 on the testset.  From this, we can estimate the expected out-of-sample error rate on a new dataset to be around 0.5% (i.e. 1-0.9945).  Which means when we apply this model on the pml-testing.csv dataset we held aside for validation, we expect only 0.5% of the 20 cases to be misclassified.             

### _Prediction on Validation Dataset_
Finally we apply our chosen prediction model to predict the 20 test cases from the original test dataset. 


```r
##first subset to exclude the unused columns (similar to what was done on the training file.) 
validation <- testing[,c(features, 'problem_id')] 

### predict
predTEST <- predict(modRF, validation) 

### print the predicted outcome
predictionResults <- data.frame(problem_id=validation$problem_id, prediction=predTEST)
print(predictionResults)
```

```
##    problem_id prediction
## 1           1          B
## 2           2          A
## 3           3          B
## 4           4          A
## 5           5          A
## 6           6          E
## 7           7          D
## 8           8          B
## 9           9          A
## 10         10          A
## 11         11          B
## 12         12          C
## 13         13          B
## 14         14          A
## 15         15          E
## 16         16          E
## 17         17          A
## 18         18          B
## 19         19          B
## 20         20          B
```

