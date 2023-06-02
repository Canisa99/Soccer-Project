df=read.csv('C:\\Users\\carlo\\Documents\\Github\\Soccer-Project\\Data\\all_157+Points',stringsAsFactors=FALSE)
df
library(glmnet)
library (car)
?sample
y=df$Points
X=data.matrix(df[,1:6])
set.seed(101)
sample <- sample.int(n = nrow(df), size = floor(.80*nrow(df)), replace = F)
dftrain <- df[sample, ]
dftest  <- df[-sample, ]
x_train <-data.matrix(dftrain[,1:6])
x_test <-data.matrix(dftest[,1:6])
y_train <- dftrain$Points
y_test <- dftest$Points
lm_fit=lm(Points~g+ga+ppda+oppda+dc+odc,data=dftrain)
summary(lm_fit)
#R-squared:  0.9596,	Adjusted R-squared:  0.9573 
#?lm
#Measures the proportion of variability in Y that can be explained using X.
summary(lm_fit)$r.sq
#The RSE provides an absolute measure of lack of fit of the model
summary(lm_fit)$sigma
?predict
##TEST##
y_predicted <- predict(lm_fit, newdata = dftest)
mean((y_predicted-y_test)^2)
sst <- sum((y_test - mean(y_test))^2)
sse <- sum((y_predicted - y_test)^2)
rsq <- 1 - sse/sst
rsq
#0.9324862

###REGRESSION DIAGNOSTIC###

#LINEARITY
lm_fit_tot=lm(Points~g+ga+ppda+oppda+dc+odc,data=df)
y_predicted_tot <- predict(lm_fit_tot)
plot(y_predicted_tot,y,xlab="Predicted values",ylab = "Actual values" )

#NORMALITY
plot(lm_fit_tot)
library("tseries")
jarque.bera.test(resid(lm_fit_tot))
resid(lm_fit_tot)

#MULTICOLLINEARITY
vif(lm_fit)
vif(lm_fit_tot)

# Cross-validation: an estimate of the test error rate using MSE
#Ideally, if we had enough data, we would set aside a validation set and use
#it to assess the performance of our prediction model. Since data are often
#scarce, this is usually not possible. To finesse the problem, K-fold crossvalidation
#uses part of the available data to fit the model, and a different
#part to test it.
library(boot)
#?glm
#glm without passing the family argument is equal to lm
?cv.glm
###LEAVE ONE OUT CROSS VALIDATION###
glm_fit <- glm(Points~g+ga+ppda+oppda+dc+odc,data=df)
cv_err <- cv.glm(df,glm_fit)#default cost=MSE and K=nrow(df) that is LOOCV
cv_err$delta #the test error estimated by the mean of the all n=nrow(df) MSE

#The vector delta has two values. Why? Which is our test error estimate?
#When K is less than the number of observations the K splits to be used are found
#by randomly partitioning the data into K groups of approximately equal size.
#In this latter case a certain amount of bias is introduced. 
#This can be reduced by using a simple adjustment.
#The second value returned in delta is the estimate adjusted by this method.

###K-FOLD CROSS VALIDATION###
set.seed(1)
cv_err_k <- cv.glm(df,glm_fit,K=8)
cv_err_k$delta
##MSE=RSS/n quindi MSE*n/n-p-1=RSE^2 da cui l'RSE###
sqrt((cv_err_k$delta)*nrow(df)/(nrow(df)-7))
#0.21924



###BEST SUBSET SELECTION###
library(leaps)
#The regsubsets() function performs best subset selection 
#by identifying the best model that contains a given number
#of predictors, where best is quantified using RSS.
?regsubsets()
regfit_full=regsubsets(Points ~ g+ga+ppda+oppda+dc+odc, df )
summary(regfit_full)
#  g   ga  ppda oppda dc  odc
#1"*" " " " "  " "   " " " "
#2"*" "*" " "  " "   " " " "
#3"*" "*" " "  "*"   " " " "
#4"*" "*" "*"  "*"   " " " "
#5"*" "*" "*"  "*"   "*" " "
#6"*" "*" "*"  "*"   "*" "*"
summary(regfit_full)$rsq
#0.7863221 0.9538611 0.9562284 0.9562556 0.9562609 0.9562609
adjr2=summary(regfit_full)$adjr2
#0.7847737 0.9531876 0.9552628 0.9549594 0.9546288 0.9542877
names=c("g","ga","ppda","oppda","dc","odc")
plot(adjr2,type="b",xlab =" Number of Variables ",ylab=" Adjusted RSq",main="Best Subset Selection")
points (3,summary(regfit_full)$adjr2[3], col ="red",cex =2, pch =20)

###FORWARD STEPWISE SELECTION###
regfit_fwd=regsubsets(Points ~ g+ga+ppda+oppda+dc+odc, data=df, method="forward" )
summary(regfit_fwd)
adjr2_fwd=summary(regfit_fwd)$adjr2
adjr2_fwd
plot(adjr2_fwd,type="b",xlab =" Number of Variables ",ylab=" Adjusted RSq",main="Forward Stepwise Selection")
points (3,summary(regfit_fwd)$adjr2[3], col ="red",cex =2, pch =20)

###BACKWARD STEPWISE SELECTION###
regfit_bwd=regsubsets(Points ~ g+ga+ppda+oppda+dc+odc, data=df, method="backward" )
summary(regfit_bwd)


###VALIDATION SET APPROACH###
regfit_best=regsubsets (Points~g+ga+ppda+oppda+dc+odc,data=dftrain)
val.errors =rep(NA ,6)
test.mat=model.matrix (Points~g+ga+ppda+oppda+dc+odc,data=dftest)
for(i in 1:6){
coefi=coef(regfit_best ,id=i)
coefi
pred=test.mat[,names(coefi)]%*%coefi
val.errors [i]= mean((y_test-pred)^2)
}

val.errors
which.min (val.errors)
#We find that the best model is the one that contains 3 variables
coef(regfit_best ,3)
#(Intercept)            g           ga        oppda 
#0.005663443  0.558129989 -0.491743670  0.055888788 

#Finally, we perform best subset selection on the full data set, and select
#the best 3-variable model. It is important that we make use of the full
#data set in order to obtain more accurate coefficient estimates.
#The best 3-variable model on the full data set 
#may differ from the corresponding model on the training set.
regfit_best=regsubsets(Points~g+ga+ppda+oppda+dc+odc,data=df)
coef(regfit_best,3)
#(Intercept)             g            ga         oppda 
#6.245005e-17  5.423079e-01 -4.971220e-01  7.104454e-02 

###CROSS VALIDATION SET APPROACH###
predict.regsubsets =function (object ,newdata ,id ,...){
form=as.formula (object$call [[2]])
mat=model.matrix (form ,newdata )
coefi =coef(object ,id=id)
xvars =names (coefi )
mat[,xvars ]%*% coefi
}
#We must perform best subset selection within each of the k training sets.
k=10
set.seed (1)
folds=sample (1:k,nrow(df),replace =TRUE)
cv.errors =matrix (NA ,k,6, dimnames =list(NULL , paste (1:6) ))
for(j in 1:k){
 best.fit =regsubsets(Points~g+ga+ppda+oppda+dc+odc,data=df[folds !=j,])
 for(i in 1:6) {
  pred=predict (best.fit ,df[folds ==j,], id=i)
  cv.errors[j,i]=mean((df$Points[folds ==j]-pred)^2)
 }
}
?apply
#Returns a vector or array or list of values obtained by applying 
#a function to margins of an array or matrix.
#for a matrix 1 indicates rows, 2 indicates columns, 
#c(1, 2) indicates rows and columns
mean.cv.errors =apply(cv.errors ,2, mean)
mean.cv.errors
library(plotrix)
standard_errors=apply(cv.errors,2,std.error)
standard_errors
par(mfrow =c(1,1))
?plot
plot(mean.cv.errors,type="b",xlab ="Size",ylab="CV error",ylim=c(0,0.3))
for(i in 1:6) {
  arrows(x0=i,y0=mean.cv.errors[i]-standard_errors[i],
         x1=i,y1=mean.cv.errors[i]+standard_errors[i],
         code=3,angle=90,length=0.15)
}
#We see that cross-validation also selects a 3-variable model but using 
#the one sigma rule the selected model is a 2-variable.
regfit_best=regsubsets(Points~g+ga+ppda+oppda+dc+odc,data=df)
coef(regfit_best,2)
