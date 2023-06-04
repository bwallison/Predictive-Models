###################
##### Library #####
###################


install.packages("FSelector")
library(FSelector)

install.packages("pROC")
library(pROC)

install.packages("glmnet")
library(glmnet)

install.packages("gbm")
library(gbm)

# install.packages("boot")
# library(boot)

install.packages("csvread")
library(csvread)

###################
## Preprocessing ##
###################

dfTest <- read.csv("ds_test.csv")
dfTraining <- read.csv("ds_training.csv")

# is.na(dfTest) is.na(dfTraining) - no N/A in data #

importanceCHI <- chi.squared(var3~., dfTraining)
importanceCFS <- cfs(var3~., dfTraining)

subsetNames <- cutoff.k.percent(importanceCHI, 0.05)

#f <- as.simple.formula(subset[2:length(subset)], "SignificantAttributes")

subsetNames <- union(subsetNames, importanceCFS)
subsetNames <- union(subsetNames, c("TARGET"))

subset <- dfTraining[, subsetNames]

idx <- sample(1:dim(subset)[1], round(0.8 * dim(subset)[1]), replace = FALSE)
trainSet <- subset[idx,]
testSet <- subset[-idx,]




###################
#### Modelling ####
###################

###################
####### GLM #######
###################
# 
# modelGLM <- glm(TARGET~., data = trainSet, family = binomial)
# 
# modelGLMPlot <- plot(modelGLM, xvar = "lambda")
# 
# predictionGLM <- predict(modelGLM, testSet, type = "response")
# 
# modelGLMCV <- cv.glm(trainSet, modelGLM)
# 
# plot.roc(testSet$TARGET, predictionGLM)
# 
# r <- trainSet$TARGET
# pi <- modelGLM$fitted.values
#
# mycost <- function(r, pi) {
#   weight1 = 1 # cost for getting 1 wrong
#   weight0 = 1 # cost for getting 0 wrong
#   c1 = (r == 1) & (pi == 0) # logical vector - true if actual 1 but predict 0
#   c0 = (r == 0) & (pi == 1) # logical vector - true if actual 0 but predict 1
#   return (mean(weight1 * c1 + weight0 * c0))
# }
#
# modelGLMCV <-  cv.glm(trainSet, modelGLM, mycost, 10)


###################
####### GBM #######
###################

modelGBM <- gbm(TARGET ~ ., data = testSet, distribution = "bernoulli", n.trees = 10, cv.folds = 10)

predictionGBM <- predict(modelGBM, testSet, 1, type = "response")

plot.roc(testSet$TARGET, predictionGBM)

###################
###### Ridge ######
###################


trainSetDataMatrix <- data.matrix(trainSet, rownames.force = NA)
testSetDataMatrix <- data.matrix(testSet, rownames.force = NA)

modelRidge <- glmnet(trainSetDataMatrix, trainSet$TARGET, family = 'binomial', alpha = 0)
#plot(modelRidge, xvar = "lambda")

modelRidgeCV <- cv.glmnet(trainSetDataMatrix, trainSet$TARGET, alpha = 0)
plot(modelRidgeCV)
lambdaValRidge <- modelRidgeCV$lambda.min

predictionRidge <- predict(modelRidge, s = lambdaValRidge, testSetDataMatrix, type = 'response')
predictionRidge <- as.vector(predictionRidge)
plot.roc(testSet$TARGET, predictionRidge)



###################
###### Lasso ######
###################


modelLasso <- glmnet(trainSetDataMatrix, trainSet$TARGET, family = 'binomial', alpha = 1)
#plot(modelLasso, xvar = "lambda")

modelLassoCV <- cv.glmnet(trainSetDataMatrix, trainSet$TARGET, alpha = 1)
plot(modelLassoCV)
lambdaValLasso <- modelLassoCV$lambda.min

predictionLasso <- predict(modelLasso, s = lambdaValLasso, testSetDataMatrix, type = 'response')
predictionLasso <- as.vector(predictionLasso)
plot.roc(testSet$TARGET, predictionLasso)

dfTest <- cbind(dfTest, dfTraining$TARGET)
names(dfTest)[names(dfTest) == 'dfTraining$TARGET'] <- 'TARGET'


subsetTest <- dfTest[, subsetNames]

subsetTest <- data.matrix(subsetTest, rownames.force = NA)
predictionOfTest <- predict(modelRidge, s = lambdaValRidge, subsetTest, type = 'response')

round <- round(as.vector(predictionOfTest), 0) 
round <- data.frame(round) 
ID <- data.frame(dfTest$ID)

dfPrediction <- data.frame(union(ID, round))
colnames(dfPrediction) <- c("ID", "TARGET")

write.csv(dfPrediction, file = "ds_sample_submission.csv", col.names = TRUE, row.names = FALSE)









   
