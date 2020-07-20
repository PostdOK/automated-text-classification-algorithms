library(tm)
library(xgboost)
library(caret)

#Data preprocessing
#
#Convert text to Corpus.
TrainingDataCorpus <- Corpus(VectorSource(TrainingData$text))
#Preprocess the corpus. In this case words are stemmed and converted to lower case. Punctuation, numbers and custmized stopwords are removed.
TrainingDataCorpus <-
  tm_map(TrainingDataCorpus, content_transformer(tolower))
TrainingDataCorpus <- tm_map(TrainingDataCorpus, removePunctuation)
TrainingDataCorpus <- tm_map(TrainingDataCorpus, stemDocument)
TrainingDataCorpus <-
  tm_map(TrainingDataCorpus, removeWords, stopwords_complete)
TrainingDataCorpus <- tm_map(TrainingDataCorpus, removeNumbers)
#Conver to document term matrix and then to a data frame that can be handeled by classification algorithms
rawDTM <- DocumentTermMatrix(TrainingDataCorpus)
DTMmatrix <- as.matrix(rawDTM)

#draw samples
set.seed(300)
id_train <- sample(1:nrow(TrainingData), round(0.75*nrow(TrainingData),0), replace = FALSE)
TrainingData_xgb <- DTMmatrix[id_train, ]
TestData_xgb <- DTMmatrix[-id_train, ]
#Get training values
TrainingValues <- TrainingData$VAR[id_train]

#Train the model and make predictions
#
#For this example, I set several tuning parameters.
#Note that colsample_bytree is set to a very low number (Kolbinger approach). This provides very good results for text classification problems!
tuning <- list(
  eta = .2,
  max_depth = 10,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 0.1
)
#Train the model to classify into your coding variable (in this case called "VAR")
XGBmodel <-
  xgboost(
    data = TrainingData_xgb,
    params = tuning,
    nrounds = 100,
    label = TrainingValues,
    verbose = 0,
    objective = "binary:logistic"
  )
#Use the model to classify the test data
predicted_class_raw <- predict(XGBmodel, TestData_xgb, type = "response")
predicted_class <- as.numeric(predicted_class_raw > 0.5)

#Evaluate the model
#
#Get real values
actual_class <- TrainingData$VAR[-id_train]
#Create contingency table and confusion matrix
conTable <- table(actual_class, predicted_class)
cm <- confusionMatrix(conTable, mode = "everything")