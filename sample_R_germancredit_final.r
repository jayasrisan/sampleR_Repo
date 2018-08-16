# Step 1: Load the dataset
credits=read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",header = FALSE,sep=" ")
head(credits)

# Step 2(a): Change the name of variables
default = credits$V21 
duration = credits$V2
amount = credits$V5
installment = credits$V8
age = credits$V13
history = factor(credits$V3)
purpose = factor(credits$V4)
rent = factor(credits$V15)

# Step 2(b)Bind the values of extracted columns in a data frame
german_credit=cbind.data.frame(default,duration,amount,purpose,installment,age,history,rent)

# Step 3: Check and see the Plot of data
str(german_credit)
plot(german_credit)
summary(german_credit)
# Step 4:  Convert the factor variables  by expanding them to a set of dummy variables variables
credit_matrix = model.matrix(default~., data=german_credit)[,-1]

# Step 5: see the structure of matrix
str(credit_matrix)
head(credit_matrix,2)  

#Step 6:change the value of defualt(rating) variable to 0 or 1 instead of 1 or 2, where 0 means good, 1 means bad

german_credit$default[german_credit$default==1]<-0 
german_credit$default[german_credit$default==2]<-1

# Step 7: Sample the data and split into ratio of 900:100
train_index = sample(1:1000,900)


#Step 8:  Split the dataset for training and for testing purpose

xtrain = credit_matrix[train_index,]

xtest = credit_matrix[-train_index,]
ytrain = german_credit$default[train_index]

ytest = german_credit$default[-train_index]

data_new=data.frame(default=ytrain,xtrain)


# Step 9: Build the Logistic regression model
LogisticModel=glm(default~., family=binomial, data=data_new)
summary(LogisticModel)

# Step 10: Evaluate the confidence interval
confint(LogisticModel)
# Step 11:  Overall effect of the rank using the wald.test function from the aod library 

library(aod)

wald.test(b=coef(LogisticModel), Sigma = vcov(LogisticModel), Terms = 6:9) # for all ranked terms for history

wald.test(b=coef(LogisticModel), Sigma = vcov(LogisticModel), Terms = 10:18) # for all ranked terms for purpose

wald.test(b=coef(LogisticModel), Sigma = vcov(LogisticModel), Terms = 19) # for the ranked term for rent

# Step 12: The Odds Ratio for each independent variable along with the 95% confidence interval for those odds ratio.


exp(coef(LogisticModel))

exp(cbind(OR=coef(LogisticModel), confint(LogisticModel))) # odds ratios adjacent  to the 95% confidence interval for odds ratios


# Step 13:  Testing :Prediction of default : credit value using  test data 

test_datas=data.frame(default=ytest,xtest)

test_data_withoutdefault=test_datas[,2:20] #removing the variable default from the data matrix

test_data_withoutdefault$defaultPrediction = predict(LogisticModel, newdata=test_data_withoutdefault, type = "response")

summary(test_datas)



# Step 14: Plotting the true positive rate against the false positive rate (ROC Curve) 


library(ROCR)

pr  = prediction(test_data_withoutdefault$defaultPrediction, test_datas$default)

prf = performance(pr, measure="tpr", x.measure="fpr")

plot(prf)

# Area under the ROC curve 

AUCLog2= performance(pr, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCLog2)












