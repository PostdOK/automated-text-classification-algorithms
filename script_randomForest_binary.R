library(tm)
library(randomForest)
library(caret)

#Data preprocessing
#
#Convert text to Corpus and the classification label (in this case called "VAR") into a factor
TrainingData$VAR <- as.factor(TrainingData$VAR)
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
DTM.df <- as.data.frame(as.matrix(rawDTM))
colnames(DTM.df) <-
  make.names(colnames(DTM.df), unique = TRUE)
TrainingDTM <- cbind(Coding = TrainingData$VAR, DTM.df)

#draw samples
id_train <-
  sample(1:nrow(TrainingDTM), round(0.75 * nrow(TrainingDTM), 0), replace = FALSE)
TestDTM <- TrainingDTM[-id_train, ]
TrainingDTM <- TrainingDTM[id_train, ]

#Train the model and make predictions
#
#Train the model to classify into your coding Variable (in this case called "VAR"). For this example, only ntree is set.
RFmodel <-
  randomForest(Coding ~ ., data = TrainingDTM, ntree = 100)
#Use the model to classify the test data
predicted_class <- predict(RFmodel, newdata = TestDTM)

#Evaluate the model
#
#Get real values
actual_class <- TestDTM$Coding
#Create contingency table and confusion matrix
conTable <- table(actual_class, predicted_class)
cm <- confusionMatrix(conTable, mode = "everything")