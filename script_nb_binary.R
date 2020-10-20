library(quanteda)
library(quanteda.textmodels)
library(caret)

#Data preprocessing
#
#Convert to corpus. Trainingdata should only include the text and the coding variable (in this case called "VAR")
TrainingDataCorpus<-corpus(TrainingData)
#different to other algorithms I recommend to split the data before preprocessing
#set seed if necessary and draw the ids for training and test data
set.seed(123)
id_train <- sample(1:nrow(TrainingData), round(0.75*nrow(TrainingData),0), replace = FALSE)
#add ids for corpus
docvars(TrainingDataCorpus, "id_numeric") <- 1:ndoc(TrainingDataCorpus)
#draw the samples and preprocess the corpora. In this case words are stemmed and converted to lower case. Punctuation and numbers are removed.
#Using quanteda, this can be done while creating a document feature matrix, which is needed for running the algorithms
TrainingDTM<-corpus_subset(TrainingDataCorpus,id_numeric %in% id_train) %>%
  dfm(stem=TRUE,remove_punct=TRUE, tolower=TRUE, remove_numbers=TRUE)
TestDTM<-corpus_subset(TrainingDataCorpus,!id_numeric %in% id_train) %>%
  dfm(stem=TRUE,remove_punct=TRUE,tolower=TRUE, remove_numbers=TRUE)

#Train the model and make predictions
#
#Train the model to classify into your coding variable (in this case called "VAR")
nbModel<-textmodel_nb(TrainingDTM,docvars(TrainingDTM,"VAR"))
#Use the model to classify the test data. dfm_match necessary to have the same features for both document-feature matrixes.
matchedDFM <- dfm_match(TestDTM, features = featnames(TrainingDTM))
predicted_class <- predict(nbModel, newdata = matchedDFM)

#Evaluate the model
#
#Get real values
actual_class <- docvars(matchedDFM, "VAR")
#Create contingency table and confusion matrix
conTable <- table(actual_class, predicted_class)
cm<-confusionMatrix(conTable,mode = "everything")
