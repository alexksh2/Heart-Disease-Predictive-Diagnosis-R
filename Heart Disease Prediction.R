# Author: Alex Khoo Shien How
# All Rights Reserved © Alex Khoo Shien How 2023
setwd("D:\\New folder\\Documents\\NTU BCG\\NTU BCG Y2S1\\CC0007 Science and Tech for Humanity\\CC0007 Proposal 2\\Heart Attack")

#https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data

## Step 1: Install external package "data.table"
# install.packages("data.table")
library(data.table)
readdataset.dt <- fread("realheart.csv", check.names = FALSE)
View(readdataset.dt)
set.seed(42)


# Check for missing values 
summary(readdataset.dt)






# Step 2: Addressing class imbalance to reduce model bias

library(ggplot2)
frequency_table <- table(readdataset.dt$target)
frequency_df_formation <- as.data.frame(frequency_table)
frequency_df_formation

ggplot(frequency_df_formation , aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  xlab("Presence of Parkinson") +
  ylab("Frequency") +
  ggtitle("Frequency Barplot for Heart Attack") +
  theme(plot.title = element_text(face="bold"))


# Result: There are 499 patients with no heart disease and 526 patients with heart disease

# Undersampling Technique
majority_class <- subset(readdataset.dt, readdataset.dt$target == 1)
minority_class <- subset(readdataset.dt, readdataset.dt$target == 0)
nrow(minority_class)
nrow(majority_class)
filtered_indices <- sample(nrow(majority_class), size = nrow(minority_class))
filtered_majority_class <- majority_class[filtered_indices,]
#View(filtered_majority_class)
new_dataset.dt <- rbind(minority_class, filtered_majority_class)



library(ggplot2)
frequency_table <- table(new_dataset.dt$target)
frequency_df_formation <- as.data.frame(frequency_table)
frequency_df_formation

ggplot(frequency_df_formation , aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  xlab("Presence of target") +
  ylab("Frequency") +
  ggtitle("Frequency Barplot for target dataset") +
  theme(plot.title = element_text(face="bold"))





# Step 3: Encode the categorical variable
new_dataset.dt$sex <- as.factor(new_dataset.dt$sex)
new_dataset.dt$cp <- as.factor(new_dataset.dt$cp)
new_dataset.dt$fbs <- as.factor(new_dataset.dt$fbs)
new_dataset.dt$restecg <- as.factor(new_dataset.dt$restecg)
new_dataset.dt$exang <- as.factor(new_dataset.dt$exang)
new_dataset.dt$slope <- as.factor(new_dataset.dt$slope)
new_dataset.dt$ca <- as.factor(new_dataset.dt$ca)
new_dataset.dt$thal <- as.factor(new_dataset.dt$thal)
new_dataset.dt$target <- as.factor(new_dataset.dt$target)

# Step 4: Splitting the dataset into Training Set and Test Set
# install.packages("caTools")
dataset.dt <- new_dataset.dt

View(dataset.dt)


library(caTools)
split = sample.split(dataset.dt$target, SplitRatio =  0.75)
training_set = subset(dataset.dt, split == TRUE)
test_set = subset(dataset.dt, split == FALSE)


View(training_set)
View(test_set)

str(dataset.dt)


# Step 5: Feature Scaling

training_set[, 1 := lapply(training_set[, 1], FUN = function(x)scale(x))]
training_set[, 4:5 := lapply(training_set[, 4:5], FUN = function(x)scale(x))]
training_set[, 8 := lapply(training_set[, 8], FUN = function(x)scale(x))]
training_set[, 10 := lapply(training_set[, 10], FUN = function(x)scale(x))]


test_set[, 1 := lapply(test_set[, 1], FUN = function(x)scale(x))]
test_set[, 4:5 := lapply(test_set[, 4:5], FUN = function(x)scale(x))]
test_set[, 8 := lapply(test_set[, 8], FUN = function(x)scale(x))]
test_set[, 10 := lapply(test_set[, 10], FUN = function(x)scale(x))]


View(training_set)










## Logistic Regression


# Fitting Logistic Regression to the Dataset (Build Logistic Regression Classifier)
classifier = glm(formula = target ~ ., family = binomial, data = training_set)


# Predicting the test results - probability of new test set
prob_pred = predict(classifier, type = "response", newdata = test_set[,-14]) #Remove the last column
prob_pred
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred


# Making the confusion matrix
actual_values <- test_set$target
class(actual_values)
y_pred <- as.factor(y_pred)
cm = table(actual_values, y_pred)
cm
# TN = 99, FP = 26, FN = 16, TN = 109

library(MLmetrics)
f1_score_LR <- F1_Score(test_set$target, y_pred, positive = "1") 
f1_score_LR #0.8384615

f2_score_LR <- FBeta_Score(test_set$target, y_pred, positive = "1", beta = 2)
f2_score_LR #0.8582677


--------------------------------------------------------------------------------------------------------
## K-NN Model
library(class)
?knn

y_pred = knn(train = training_set[, -14], 
             test = test_set[, -14],
             cl = training_set$target,
             k = 5)
y_pred



# Making the confusion matrix
actual_values <- test_set$target
cm = table(actual_values, y_pred)
cm
# TN = 111, FP = 14, FN = 25, TP = 100

library(MLmetrics)
f1_score_KNN <- F1_Score(test_set$target, y_pred, positive = "1") 

f1_score_KNN #0.8368201

f2_score_KNN <- FBeta_Score(test_set$target, y_pred, positive = "1", beta = 2)
f2_score_KNN #0.8143322
 
-----------------------------------------------------------------------------------------
## Decision Tree Classification
  
  
# install.packages("rpart")
library(rpart)
classifier <- rpart(formula = target ~ ., 
                    data = training_set, method = 'class',
                    control = rpart.control(minsplit = 2, cp = 0))

summary(classifier)

# Step 1: Plot the Maximal Tree and the results
# install.packages("rpart.plot")
#library("rpart.plot")
#rpart.plot(classifier, nn = T, main = "Maximal Tree") 


# Step 2: Display the pruning sequence and 10-fold CV errors
plotcp(classifier)

# Step 3: Print out the pruning sequence and 10-fold CV errors as a table
printcp(classifier)

# Step 4: Find out the most important variable and plot the bar chart of variable importance
classifier$variable.importance



var_importance <- classifier$variable.importance
var_importance
sorted_var_importance <- var_importance[order(var_importance, decreasing = TRUE)]
sorted_var_importance
rownames <- colnames(classifier$variable.importance)
rownames

barplot(sorted_var_importance, 
        names.arg = names(classifier$variable.importance),
        xlab = "Variable Importance",
        #ylab = "Variable",
        col = "darkblue",  # Change the color as needed
        horiz = TRUE,
        las = 2,
        main = "Variable Importance Bar Chart (Maximal Tree CART Model)") 
par(mar = c(5.1,15,4.1,2.1)) # bottom, left, top right


# Step 5: Extract the optimal tree
# Compute min CVerror + 1SE in maximal tree

classifier$cptable
?which.min()
CVerror.xerror <- classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xerror"]
CVerror.xerror

CVerror.std <- classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xstd"]
CVerror.std

CVerror.cap <- classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xerror"] + classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xstd"]
CVerror.cap




# Step 6: Find the most optimal CP region whose CV error is just below CVerror.cap in maximal tree
i <- 1
j <- 4

while(classifier$cptable[i,j] > CVerror.cap){
  i <- i + 1
}

# Step 7: Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split
cp.opt <-  ifelse(i > 1, sqrt(classifier$cptable[i,1] * classifier$cptable[i-1,1]),1)
cp.opt


# Step 8: Get the optimal tree 
classifier2 <- prune(classifier, cp = cp.opt)
printcp(classifier2, digits = 3)
?printcp()


# Step 9: Plot the CART model and corresponding variable importance bar chart
library("rpart.plot")
rpart.plot(classifier2, nn = T, main = "Optimal CART Model for Predictive Diagnosis of Heart Disease") 

# Step 10: Print the summary of CART model and the CART model
print(classifier2)
summary(classifier2)





# Step 11: Print the variable importance bar chart
var_importance <- classifier2$variable.importance
var_importance
sorted_var_importance <- var_importance[order(var_importance, decreasing = TRUE)]
sorted_var_importance
rownames <- colnames(classifier2$variable.importance)
rownames

barplot(sorted_var_importance, 
        names.arg = names(classifier2$variable.importance),
        xlab = "Variable Importance",
        #ylab = "Variable",
        col = "darkblue",  # Change the color as needed
        horiz = TRUE,
        las = 2,
        main = "Variable Importance Bar Chart (Optimal CART Model)") 
par(mar = c(5.1,15,4.1,2.1)) # bottom, left, top right



# Step 12: Checking prediction accuracy by making the confusion matrix table
classifier2.predict <- predict(classifier2, newdata = test_set[,-14], type = "class")
classifier2.predict

results <- data.frame(test_set, classifier2.predict)
results

cm = table(results$target, results$classifier2.predict)
cm # TN = 117, FP = 8, FN = 8, TP = 117

# Step 13: Check the f1 score of the predicted results
# install.packages("MLmetrics")
library(MLmetrics)

f1_score_CT <- F1_Score(test_set$target, classifier2.predict, positive = "1") 
f1_score_CT #0.936

f2_score_CT <- FBeta_Score(test_set$target, classifier2.predict, positive = "1", beta = 2)
f2_score_CT #0.936

# Step 14: Predict the class probability 
classifier2.predictprob <- predict(classifier2, newdata = test_set[,-14], type = "prob")
classifier2.predictprob


------------------------------------------------------------------------------
  
# Random Forest Model
library(randomForest)

classifier3 = randomForest(x = training_set[,-14], y = training_set$target, ntree = 30, keep.forest = TRUE) #Note ntree large will cause overfitting

# Step 1: Predict the test results - probability of new test set
classifier3.predict = predict(classifier3, newdata = test_set[,-14]) 
classifier3.predict


# Step 2: Predict the class probability
classifier3.predict_prob = predict(classifier3, newdata = test_set[,-14], type = "prob") 
classifier3.predict_prob

# Step 3: Plot confusion matrix
cm = table(test_set$target, classifier3.predict)
cm #TN = 123, FP = 2, FN = 4, TP = 121



# Step 4: Plot a variable importance plot
varImpPlot(classifier3, main = "Variable Importance Plot of Random Forest")


library(MLmetrics)

f1_score_RF <- F1_Score(test_set$target, classifier3.predict, positive = "1") 
f1_score_RF #0.9758065

f2_score_RF <- FBeta_Score(test_set$target, classifier3.predict, positive = "1", beta = 2)
f2_score_RF #0.9711075
