library(tm)
library(caret)
library(irr)

#Data preprocessing
#
#Convert text to Corpus and the classification label (in this case called "Sentiment") into a factor
TrainingData$Sentiment <- as.factor(TrainingData$Sentiment)
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
DTM.df <- as.data.frame(as.matrix(DocumentTermMatrix(TrainingDataCorpus)))
TrainingDTM <- cbind(Coding = TrainingData$Sentiment, DTM.df)

#draw samples
id_train <-
  sample(1:nrow(TrainingDTM), round(0.75 * nrow(TrainingDTM), 0), replace = FALSE)
TestDTM <- TrainingDTM[-id_train, ]
TrainingDTM <- TrainingDTM[id_train, ]

#Train the model and make predictions
#
#Train the model to classify into your coding Variable (in this case called "Sentiment")
SVMmodel <-
  train(Coding ~ ., data = TrainingDTM, method = 'svmLinear3')
#Use the model to classify the test data. na.omit removes rows with missing values (if any)
predicted_class <- predict(SVMmodel, na.omit(TestDTM))

#Evaluate the model
#
#Get real values
actual_class <- TestDTM$Coding
#Create contingency table and confusion matrix
cm <- confusionMatrix(table(actual_class, predicted_class), mode = "everything")
#Calculate Krippendorffs Alpha, especially for ordered classes
for_kripp<-as.matrix(rbind(predicted_class,actual_class))
krippends<-kripp.alpha(for_kripp,'ordinal')
