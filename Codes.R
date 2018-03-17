

library(caret)
library(RTextTools)

AllData <- read.csv('train.csv')

str(AllData)

# Data Pre-processing

AllData$Label = rowSums(AllData[,3:ncol(AllData)])

AllData$Label[AllData$Label!=0] <- 1

sum(AllData$Label==1)

sum(AllData$Label==0)

AllData1 = AllData[,c(2,9)]

# Down-sampling and train-test splitting

AllData1$Label <- as.factor(AllData1$Label)
AllData1 <- downSample(AllData1$comment_text,AllData1$Label)


colnames(AllData1) <- c("comment_text", "Label")



inTrainingSet=createDataPartition(AllData1$Label, p=0.80, list=FALSE)
# View(inTrainingSet)


TrainSet=AllData1[inTrainingSet,]
TestSet=AllData1[-inTrainingSet,]
# nrow(TrainSet)


AllData2 = rbind(TrainSet,TestSet)



# Creating a term-document matrix

doc_matrix <- create_matrix(AllData2$comment_text, language="english", 
                            #removeNumbers=TRUE,
                            stemWords=TRUE, 
                            ngramLength = 1,
                            minWordLength =3,
                            removeSparseTerms=0.9,
                            weighting = tm::weightTfIdf)

# Creating a container

container <- create_container(doc_matrix, AllData2$Label, 
                              trainSize=1:nrow(TrainSet),
                              testSize=(nrow(TrainSet)+1):
                                nrow(AllData2),
                              virgin=FALSE)


# Creating multiple training models

# library(GLMNET)
# library(maxent)
# library(caTools)
# library(ipred)
# library(randomForest)
# library(nnet)
# library(tree)


SVM <- train_model(container,"SVM")
GLMNET <- train_model(container,"GLMNET")
MAXENT <- train_model(container,"MAXENT")
SLDA <- train_model(container,"SLDA")
BOOSTING <- train_model(container,"BOOSTING")
BAGGING <- train_model(container,"BAGGING")
RF <- train_model(container,"RF")
# NNET <- train_model(container,"NNET")
TREE <- train_model(container,"TREE")


# Creating respective classifiers

SVM_CLASSIFY <- classify_model(container, SVM)
GLMNET_CLASSIFY <- classify_model(container, GLMNET)
MAXENT_CLASSIFY <- classify_model(container, MAXENT)
SLDA_CLASSIFY <- classify_model(container, SLDA)
BOOSTING_CLASSIFY <- classify_model(container, BOOSTING)
BAGGING_CLASSIFY <- classify_model(container, BAGGING)
RF_CLASSIFY <- classify_model(container, RF)
# NNET_CLASSIFY <- classify_model(container, NNET)
TREE_CLASSIFY <- classify_model(container, TREE)


# Analytics

analytics <- create_analytics(container,
                              cbind(
                                SVM_CLASSIFY, 
                                GLMNET_CLASSIFY,
                                MAXENT_CLASSIFY,
                                SLDA_CLASSIFY,
                                BOOSTING_CLASSIFY, 
                                BAGGING_CLASSIFY,
                                RF_CLASSIFY,
                                #NNET_CLASSIFY, 
                                TREE_CLASSIFY
                              ))

View(analytics@ensemble_summary)
document_summary <- as.data.frame (analytics@document_summary)
sum(document_summary$PROBABILITY_INCORRECT)

confusionMatrix(document_summary$MANUAL_CODE, document_summary$CONSENSUS_CODE)

confusionMatrix(document_summary$MANUAL_CODE, document_summary$PROBABILITY_CODE)


score_analytics <- create_scoreSummary(container, 
                                       cbind(
                                         SVM_CLASSIFY, 
                                         GLMNET_CLASSIFY,
                                         MAXENT_CLASSIFY,
                                         SLDA_CLASSIFY,
                                         BOOSTING_CLASSIFY, 
                                         BAGGING_CLASSIFY,
                                         RF_CLASSIFY,
                                         # NNET_CLASSIFY, 
                                         TREE_CLASSIFY
                                       ))

PRSummary <- create_precisionRecallSummary(container,
                                           cbind(
                                             SVM_CLASSIFY, 
                                             GLMNET_CLASSIFY,
                                             MAXENT_CLASSIFY,
                                             SLDA_CLASSIFY,
                                             BOOSTING_CLASSIFY, 
                                             BAGGING_CLASSIFY,
                                             RF_CLASSIFY,
                                             # NNET_CLASSIFY, 
                                             TREE_CLASSIFY
                                           ))                                        

summary(analytics)
topic_summary <- analytics@label_summary
alg_summary <- analytics@algorithm_summary
ens_summary <-analytics@ensemble_summary
doc_summary <- analytics@document_summary


create_ensembleSummary(analytics@document_summary)

write.csv(analytics@document_summary, "DocumentSummary.csv")
