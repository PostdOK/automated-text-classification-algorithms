library(tm)
library(xgboost)
library(caret)
library(irr)

#Data preprocessing
#
#Convert text to Corpus
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
DTMmatrix <- as.matrix(DocumentTermMatrix(TrainingDataCorpus))

#draw samples
set.seed(300)
id_train <- sample(1:nrow(TrainingData), round(0.75*nrow(TrainingData),0), replace = FALSE)
TrainingData_xgb <- DTMmatrix[id_train, ]
TestData_xgb <- DTMmatrix[-id_train, ]
#Get training values. If the variable includes negative values, it needs to be transformed (in this case min is -1)
TrainingValues <- TrainingData$Sentiment[id_train]+1

#Train the model and make predictions
#
#For this example, I set several tuning parameters.
#Note that colsample_bytree is set to a very low number (Kolbinger approach). This provides very good results for text classification problems!
tuning <- list(
  eta = .2,
  max_depth = 10,
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
    objective = "multi:softprob",
    num_class = 3
  )
#Use the model to classify the test data. Here, the class with the highest probability gets extracted.
#Here, the actual values for each prediction are added as well
predicted_class_raw <- predict(XGBmodel, TestData_xgb, type = "response")
predicted_class_matrix<-t(matrix(predicted_class_raw,
       nrow = 3,
       ncol = length(predicted_class_raw) / 3))
predicted_class<-max.col(predicted_class_matrix)-2

#Evaluate the model
#
#Get real values
actual_class <- TrainingData$Sentiment[-id_train]
#Create contingency table and confusion matrix
cm<-confusionMatrix(table(actual_class,predicted_class), mode = "everything")
#Calculate Krippendorffs Alpha, especially for ordered classes
for_kripp<-as.matrix(rbind(predicted_class$label,actual_class$max_prob))
krippends<-kripp.alpha(for_kripp,'ordinal')
