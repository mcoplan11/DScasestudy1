#DScasestudy
getwd()
setwd("/Users/mitchell.coplan/Downloads")
data <- read.table("DScasestudy1.txt", header = T, sep="\t")
nrow(data)
ncol(data)

#This is a difficult dataset to work with bc it has many features and a small amount of samples

sum(data$response) #123 postive out of 530 (23%) 

#get rid of some noise
#throw out all the features that are all 0 or all 1
t <- colSums(data)
tt <- t[t==530|t==0]
tt <- as.data.frame(tt)
tt <- row.names(tt)
tt <- as.vector(tt)
data1 <- data[, -which(names(data) %in% tt)]

#View some of the data
test1 <- data1[1:200,1:50]
View(test1)


#######Random Forest model############

#set training and testing data#
train=sample(1:nrow(data1),350)

#mtry is num of Variables randomly chosen at each split
ntry = seq(from=10 ,to=120, by=10) #no of trees-a vector of 100 values 
oob.err=double(12)
test.err=double(12)

for(mtry in ntry) 
{
  rf=randomForest(response ~ . , data = data1 , subset = train,mtry=mtry,ntree=500) 
  oob.err[mtry] = rf$mse[500] #Error of all Trees fitted
  
  pred<-predict(rf,data1[-train,]) #Predictions on Test Set for each Tree
  test.err[mtry]= with(data1[-train,], mean( (response - pred)^2)) #Mean Squared Test Error
  
  cat(mtry," ") #printing the output to the console
  
}

matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))
#Error seems to minimize with mtry = 90

rf=randomForest(response ~ . , data = data1 , subset = train,mtry=90,ntree=500) 
## R squared value of .3019
## Mean of squared residuals: 0.1230947
## test error 0.1299957


test.err= with(data1[-train,], mean( (response - pred)^2)) #Mean Squared Test Error


##this model skews the preditions so that a 0 is more likely than a 1.(i.e. lots of type 2 errors) 
# this probably because there is is only ~20% postives in our dataset


#####Gradient boosting model#######
library(gbm)
train=sample(1:530,size=400)
data1$response
data1.boost=gbm(response ~ . ,data = data1[train,],distribution = "gaussian",n.trees = 10000,
                 shrinkage = 0.01, interaction.depth = 4)
data1.boost

summary(data1.boost) #Summary gives a table of Variable Importance and a plot of Variable Importance

#Plot of Response variable with V15261 variable

plot(data1.boost,i="V15261") 
plot(data1.boost,i="V9867") 

#as the average number of rooms increases the the price increases
cor(data1$response,data1$V15261)

n.trees = seq(from=100 ,to=10000, by=100) #no of trees-a vector of 100 values 

#Generating a Prediction matrix for each Tree
predmatrix<-predict(data1.boost,data1[-train,],n.trees = n.trees)
dim(predmatrix) #dimentions of the Prediction Matrix
View(predmatrix)
#Calculating The Mean squared Test Error
test.error<-with(data1[-train,],apply( (predmatrix-response)^2,2,mean))
head(test.error) #contains the Mean squared test error for each of the 100 trees averaged
#test error around 10.8%

#Plotting the test error vs number of trees

plot(n.trees , test.error , pch=19,col="blue",xlab="Number of Trees",ylab="Test Error", main = "Perfomance of Boosting on Test Set")

#adding the RandomForests Minimum Error line trained on same data and similar parameters
abline(h = min(test.err),col="red") #test.err is the test error of a Random forest fitted on same data
legend("topright",c("Minimum Test error Line for Random Forests"),col="red",lty=1,lwd=1)



#####LASSO########
library(glmnet)
data1[,"train"] <- ifelse(runif(nrow(data1))<0.8,1,0)
#separate training and test sets
trainset <- data1[data1$train==1,]
testset <- data1[data1$train==0,]
#get column index of train flag
trainColNum <- grep("train",names(trainset))
#remove train flag column from train and test sets
trainset <- trainset[,-trainColNum]
testset <- testset[,-trainColNum]

x <- model.matrix(response~.,trainset)
#convert class to numerical variable
y <- trainset$response

#perform grid search to find optimal value of lambda
#family= binomial => logistic regression, alpha=1 => lasso
# check docs to explore other type.measure options
cv.out <- cv.glmnet(x,y,alpha=1,family='binomial',type.measure = 'mse' )
#plot result
plot(cv.out)

#min value of lambda
lambda_min <- cv.out$lambda.min
#best value of lambda
lambda_1se <- cv.out$lambda.1se
#regression coefficients
co <- coef(cv.out,s=lambda_1se)
data.frame(name = co@Dimnames[[1]][co@i + 1], coefficient = co@x)

#name coefficient
#1 (Intercept)  -1.5633868
#2       V9867   0.2322918
#3      V15261   1.8722575


#get test data
x_test <- model.matrix(response~.,testset)
#predict class, type=”class”
lasso_prob <- predict(cv.out,newx = x_test,s=lambda_1se,type='response')
#translate probabilities to predictions
lasso_predict <- rep('neg',nrow(testset))
lasso_predict[lasso_prob>.5] <- 'pos'
#confusion matrix
table(pred=lasso_predict,true=testset$response)
test.err= mean( (testset$response - lasso_prob)^2) #Mean Squared Test Error
#test.err = 0.09609721

#The bias-variance tradeoff tells us that the simpler function should be preferred
#because it is less likely to overfit the training data.


##some thoughts on the data.... 
#1.  What is the goal to of this model?  Is it better to get false negatives or
# or false positives.  Depending on this answer, I would have to change some parartmeters of the model
#in order to change the sensitivity and specificity

#2.  Is it be beneficail to identify a small number of varibles that are the most
# predicitve and use those feature to create a model?   


