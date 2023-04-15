#BIOSTAT 626 MID1
#Name:Xuheng Qiang
#data preprocessing
library(dplyr)
library(ISLR)
library(MASS)
library(e1071)
library(neuralnet)
library(caret)
library(adabag)
library(keras)
library(tensorflow)
set.seed(42406815)
data<-read.table("C:/Users/qiang/OneDrive/Desktop/BIOS 626/mid/training_data.txt",header=TRUE)
datatest<-read.table("C:/Users/qiang/OneDrive/Desktop/BIOS 626/mid/test_data.txt",header=TRUE)

n <- nrow(data)
data$bin<-0
for(i in 1:nrow(data))
{
  if(data$activity[i]<4)
  {
    data$bin[i]<-1
  }
}
attach(data)
standardized.X=scale(data[,-c(564,1,2)])
test=1:1000
train.X=standardized.X[-test,]
test.X=standardized.X[test,]
train.Y=bin[-test]
test.Y=bin[test]
traindata<-cbind(train.X,train.Y)
traindata<-as.data.frame(traindata)
testdata<-cbind(test.X,test.Y)
testdata<-as.data.frame(testdata)

for(i in 1:nrow(data))
{
  if(data$activity[i]>6)
  {
    data$mul[i]<-7
  }else
  {
    data$mul[i]<-data$activity[i]
  }
}
train.Y2=data$mul[-test]
test.Y2=data$mul[test]
traindata2<-cbind(train.X,train.Y2)
traindata2<-as.data.frame(traindata2)
testdata2<-cbind(test.X,test.Y2)
testdata2<-as.data.frame(testdata2)

################################################################################
#binary
################################################################################
#LOGISTIC
glm.fit=glm(as.factor(train.Y)~., family=binomial(link=logit), data=traindata)
#does not converge

#LDA
lda<-lda(as.factor(train.Y)~., data=traindata)
#variables are collinear

#QDA
qda.fit = qda(as.factor(train.Y)~., data=traindata)
#some group is too small for 'qda'

#SVM with Linear Kernel
svm.lin.fit = svm(as.factor(train.Y)~., data=traindata,
                  kernel="linear", scale=F, cost=10)
svm.lin.pred = predict(svm.lin.fit, testdata)
table(svm.lin.pred, test.Y)
mean(svm.lin.pred == as.numeric(test.Y)) #1

#SVM with radial kernel
svm.rad.fit<-svm(as.factor(train.Y)~., data=traindata, kernel="radial", cost=10, scale=FALSE)
svm.rad.pred = predict(svm.rad.fit, testdata)
table(svm.rad.pred, test.Y)
mean(svm.rad.pred == as.numeric(test.Y)) #1

# SVM lin 5-folds cv
svmLModel <- caret::train(as.factor(train.Y) ~ ., data = traindata, method = "svmLinear", 
                   trControl = trainControl(method = "cv", number = 5))
svmLPred <- predict(svmLModel, testdata)
table(svmLPred, test.Y)
mean(svmLPred == as.numeric(test.Y)) #1

# SVM rad 5 folds cv
svmRModel <- caret::train(as.factor(train.Y) ~ ., data = traindata, method = "svmRadial", 
                   trControl = trainControl(method = "cv", number = 5))
svmRPred <- predict(svmRModel, testdata)
table(svmRPred, test.Y)
mean(svmRPred == as.numeric(test.Y)) #1

#Single Hidden-layer Neural Network
nn_fit = neuralnet(as.factor(train.Y)~., data=traindata,
                   hidden = 3, act.fct = "logistic",
                   linear.output=F)
nn.test = neuralnet::compute(nn_fit, testdata)
nn.prob = nn.test$net.result[,2]
nn.pred = (nn.prob>=0.5)+0
table(nn.pred, test.Y)
mean(nn.pred==as.numeric(test.Y)) #1

#AdaBoost
traindata$train.Y<-as.factor(traindata$train.Y)
ada.fit <- boosting(train.Y~., data=traindata,mfinal=4)
ada.prob = predict(ada.fit, testdata)$prob[,2]
ada.pred = rep(0,length(ada.prob))
ada.pred[ada.prob>=0.5]=1
table(ada.pred, test.Y) 
mean(ada.pred==as.numeric(test.Y)) #0.999

#community voting
vote_vec = as.numeric(svmLPred) + nn.pred + ada.pred
ensemble.pred = rep(0, length(vote_vec))
ensemble.pred[vote_vec>=2] = 1
table(ensemble.pred, test.Y)
mean(ensemble.pred==as.numeric(test.Y)) #1 (1, 0.999)

################################################################################
#multi-class
################################################################################
#LDA
lda2<-lda(as.factor(train.Y2)~., data=traindata2)
#variables are collinear

#QDA
qda.fit2 = qda(as.factor(train.Y2)~., data=traindata2)
#some group is too small for 'qda'

#SVM with Linear Kernel
svm.lin.fit2 = svm(as.factor(train.Y2)~., data=traindata2,
                  kernel="linear", scale=F, cost=10)
svm.lin.pred2 = predict(svm.lin.fit2, testdata2)
table(svm.lin.pred2, test.Y2)
mean(svm.lin.pred2 == as.numeric(test.Y2)) #0.949

#SVM with radial kernel
svm.rad.fit2<-svm(as.factor(train.Y2)~., data=traindata2, kernel="radial", cost=10, scale=FALSE)
svm.rad.pred2 = predict(svm.rad.fit2, testdata2)
table(svm.rad.pred2, test.Y2)
mean(svm.rad.pred2 == as.numeric(test.Y2)) #0.959

# SVM lin 5-folds cv
svmLModel2 <- caret::train(as.factor(train.Y2) ~ ., data = traindata2, method = "svmLinear", 
                   trControl = trainControl(method = "cv", number = 5))
svmLPred2 <- predict(svmLModel2, testdata2)
table(svmLPred2, test.Y2)
mean(svmLPred2 == as.numeric(test.Y2)) #0.947

# SVM rad 5 folds cv
svmRModel2 <- caret::train(as.factor(train.Y2) ~ ., data = traindata2, method = "svmRadial", 
                   trControl = trainControl(method = "cv", number = 5))
svmRPred2 <- predict(svmRModel2, testdata2)
table(svmRPred2, test.Y2)
mean(svmRPred2 == as.numeric(test.Y2)) #0.958

#Single Hidden-layer Neural Network
nn.fit2 = neuralnet(as.factor(train.Y2)~., data=traindata2,
                   hidden = 3, act.fct = "softmax",
                   linear.output=F)
nn.test2 = compute(nn.fit2, testdata2)
nn.prob2 <- nn.test2$net.result
nn.pred2 <- max.col(nn.prob2)
table(nn.pred2, test.Y2)
mean(nn.pred2==as.numeric(test.Y2)) #0.90+

#Neural Network with Keras
train.Y2alt<-to_categorical(data$mul[-test]-1, num_classes = 7)
test.Y2alt<-to_categorical(data$mul[test]-1, num_classes = 7)

model <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = 561) %>% 
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 7, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c('accuracy')
)

early_stopping <- callback_early_stopping(patience = 5, restore_best_weights = TRUE)

result<-c()
for(i in 1:10)
{
  model %>% fit(
    x=train.X,
    y=train.Y2alt,
    epochs = 50,
    batch_size = 32,
    validation_split = 0.2,
    callbacks = list(early_stopping)
  )
  
  r<- model %>% evaluate(test.X, test.Y2alt)
  result<-cbind(result, as.numeric(r[2]))
}

mean(result)  
#0.954 learning rate=0.001 256 neurons first layer
#0.9526 learning rate=0.01
#0.9469 learning rate=0.1

#0.9487 learning rate=0.001 512 neurons first layer
#0.9464 learning rate=0.01
#0.9415 learning rate=0.1



#AdaBoost
traindata2$train.Y2<-as.factor(traindata2$train.Y2)
ada.fit2 <- boosting(train.Y2~., data=traindata2,mfinal=4)
ada.prob2 = predict(ada.fit2, testdata2)$prob
ada.pred2 <- max.col(ada.prob2)
table(ada.pred2, test.Y2)
mean(ada.pred2==as.numeric(test.Y2)) #0.88+

#community voting
vote_vec2 = as.numeric(svmRPred2) + nn.pred2 + ada.pred2
ensemble.pred2 = rep(0, length(vote_vec2))
ensemble.pred2 <- apply(cbind(svmRPred2, as.factor(nn.pred2),as.factor(ada.pred2))
                        , 1, function(x) names(which.max(table(x))))
table(ensemble.pred2, test.Y2)
mean(ensemble.pred2==as.numeric(test.Y2)) #0.953 (0.922，0.882)

################################################################################
#test data results
################################################################################
datatest<-read.table("C:/Users/qiang/OneDrive/Desktop/BIOS 626/mid/test_data.txt",header=TRUE)
datatest=scale(datatest[,-1])

#binary:
svmLPred0 <- predict(svmLModel, datatest)
nn.test0 = compute(nn_fit, datatest)
nn.prob0 <- nn.test0$net.result
nn.pred0 <- max.col(nn.prob0)
ada.prob0 = predict(ada.fit, as.data.frame(datatest))$prob
ada.pred0 <- max.col(ada.prob0)
vote_vec0 = as.numeric(svmLPred0) + nn.pred0 + ada.pred0-3
ensemble.pred0 = rep(0, length(vote_vec0))
ensemble.pred0[vote_vec0>=2] = 1
table(ensemble.pred0)
write.table(ensemble.pred0, "binary_6815.txt", col.names = FALSE, row.names = FALSE)

#multi-class: submit 1: ensemble method
#(majority voting by 5-fold cv svm radial, single layer nn, adaboost)
svmRPred0 <- predict(svmRModel2, datatest)
nn.test0 = compute(nn.fit2, datatest)
nn.prob0 <- nn.test0$net.result
nn.pred0 <- max.col(nn.prob0)
ada.prob0 = predict(ada.fit2, as.data.frame(datatest))$prob
ada.pred0 <- max.col(ada.prob0)
vote_vec0 = as.numeric(svmRPred0) + nn.pred0 + ada.pred0
ensemble.pred0 = rep(0, length(vote_vec0))
ensemble.pred0 <- apply(cbind(svmRPred0, as.factor(nn.pred0),as.factor(ada.pred0))
                        , 1, function(x) names(which.max(table(x))))
table(ensemble.pred0)
outm<-as.numeric(ensemble.pred0)
table(outm)
write.table(outm, "multiclass_6815.txt", col.names = FALSE, row.names = FALSE)

#multi-class: submit 1: multi-layer neural network
mnnpred <- model %>% predict(datatest)
outm2<-max.col(mnnpred, ties.method = "first")
table(outm2)
write.table(outm2, "multiclass_6815.txt", col.names = FALSE, row.names = FALSE)










