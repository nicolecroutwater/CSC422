# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:49:35 2017

@author: jtwall3

This is my inital data exploration and model building script. I did a lot of data exploration in the console that is
not recorded here. Things that were not necessary to the exploration I removed from my script so it is more polished.
"""

import pandas as pd
import numpy as np
import sklearn.linear_model as glm
import sklearn.svm as svms
import sklearn.tree as trees
import sklearn.ensemble as ens
import sklearn.neighbors as neigh
import sklearn.naive_bayes as nbc
import sklearn.metrics as met
import sklearn.model_selection as ms
from keras.models import Sequential
from keras.layers import Dense
import sklearn.datasets as ds
import scipy.sparse
import matplotlib.pyplot as pp

#I am using numpy.float32 instead of the default numpy.float64 bc the values in the dataset are 6 decimal places and numpy.float32
#still has good accuracy at 6 decimal places and it also cuts memory usage by half. Make the datasets a little easier to manipulate.
#This may cause problems though so I may go back to np.float64
variables, truth = ds.load_svmlight_file("C:/Users/jtwall/Files/5Jr/CSC422/Project/TV_News_Channel_Commercial_Detection_Dataset/BBC.txt", dtype = np.float32)

#The following is an outtake from the description of the dataset. I have made some corrections out to the side because I have
#a strong suspicion that the description is wrong in a few cases.

"""
The Feature File is represented in Lib SVM data format and contains approximetly 63% commercial instances( Positives).
Dimension index for different Features are as Follows :
Labels :
+1/-1 ( Commercials/Non Commercials)
Feature:
Dimension Index in feature File

*****I have subtracted one from each of the indexes from the readme since my DataFrame is zero index*****

Shot Length
0
Motion Distribution ( Mean and Variance)
1 - 2
Frame Difference Distribution ( Mean and Variance)
3 - 4
Short time energy ( Mean and Variance)
5 - 6
ZCR( Mean and Variance)
7 - 8
Spectral Centroid ( Mean and Variance)
9 - 10
Spectral Roll off ( Mean and Variance)
11 - 12
Spectral Flux ( Mean and Variance)
13 - 14
Fundamental Frequency ( Mean and Variance)
15 - 16
Motion Distribution ( 40 bins)
17 - 57                                                            I think this is actually 17-56
Frame Difference Distribution ( 32 bins)
58 - 90                                                            I think this is actually 57-88
Text area distribution ( 15 bins Mean and 15 bins for variance )
91 - 121                                                           I think this is actually 89-118
Bag of Audio Words ( 4000 bins)
122 - 4122                                                         I think this is actually 119-4122
Edge change Ratio ( Mean and Variance)
4123 - 4124
"""

#This is ensure that the truth values are binary, 0 and 1, not -1 and 1
prey = pd.DataFrame(truth)

y = []

for index, row in prey.iterrows():
    if (row[0] == -1):
        y.append(0)
    else:
        y.append(1)

y = pd.DataFrame(y)

sparse = pd.DataFrame(variables.toarray())

#Separating out the dense columns
dense = sparse[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,4123,4124]]

dense.columns = ["Length", "MD_M", "MD_V", "FDD_M", "FDD_V", "STE_M", "STE_V", "ZCR_M", "ZCR_V",
            "SC_M", "SC_V", "SRO_M", "SRO_V", "SF_M", "SF_V", "FF_M", "FF_V", "ECR_M", "ECR_V"]

#This is to drop missing data
missingrows = []

for index, row in dense.iterrows():
    if (row["FDD_M"] == 0):
        missingrows.append(index)

dense.drop(missingrows)

#Separating out the different sparse parts of the matrix
md = sparse[list(range(17,57))]
fdd = sparse[list(range(57,89))]
tad = sparse[list(range(89,119))]
bow = sparse[list(range(119,4123))]

#Obtaining summary stats for each sparse part of the matrix
dense["MD_N"] = md.astype(bool).sum(axis=1)
dense["FDD_N"] = fdd.astype(bool).sum(axis=1)
dense["TAD_N"] = tad.astype(bool).sum(axis=1)/2
dense["TAD_S"] = tad.sum(axis=1)
dense["BOW_N"] = bow.astype(bool).sum(axis=1)

#split the data into train test split for intial model building
dense_train, dense_test, y_train, y_test = ms.train_test_split(dense, y, test_size = .33, random_state = 12)

y_train = y_train[0]
y_test = y_test[0]

#These are all of my initial models that I built. I record the metrics for each one then graph them to figure out
#Which ones did the best

#Linear Models
lrl1 = glm.LogisticRegression(penalty = "l1", solver = "liblinear")
lrl1.fit(dense_train, y_train)
lrl1_yhat = lrl1.predict(dense_test)
lrl1_yhatprob = lrl1.predict_proba(dense_test)[:,1]
print("Logistic regression with an l1 penalty")
print("Accuracy = " + str(met.accuracy_score(y_test, lrl1_yhat)))
print("Recall = " + str(met.recall_score(y_test, lrl1_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, lrl1_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, lrl1_yhat)))
print("AUROC = " + str(met.roc_auc_score(y_test, lrl1_yhatprob)))


lrl2 = glm.LogisticRegression(penalty = "l2")
lrl2.fit(dense_train, y_train)
lrl2_yhat = lrl2.predict(dense_test)
lrl2_yhatprob = lrl2.predict_proba(dense_test)[:,1]
print("\nLogistic regression with an l2 penalty")
print("Accuracy = " + str(met.accuracy_score(y_test, lrl2_yhat)))
print("Recall = " + str(met.recall_score(y_test, lrl2_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, lrl2_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, lrl2_yhat)))
print("AUROC = " + str(met.roc_auc_score(y_test, lrl2_yhatprob)))


sgdc = glm.SGDClassifier()
sgdc.fit(dense_train, y_train)
sgdc_yhat = sgdc.predict(dense_test)
print("\nStochastic Gradient Descent")
print("Accuracy = " + str(met.accuracy_score(y_test, sgdc_yhat)))
print("Recall = " + str(met.recall_score(y_test, sgdc_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, sgdc_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, sgdc_yhat)))

#Neighbor Models
knn = neigh.KNeighborsClassifier(n_neighbors = 20)
knn.fit(dense_train, y_train)
knn_yhat = knn.predict(dense_test)
knn_yhatprob = knn.predict_proba(dense_test)[:,1]
print("\nKNN")
print("Accuracy = " + str(met.accuracy_score(y_test, knn_yhat)))
print("Recall = " + str(met.recall_score(y_test, knn_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, knn_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, knn_yhat)))
print("AUROC = " + str(met.roc_auc_score(y_test, knn_yhatprob)))


#Tree Based Models
dt = trees.DecisionTreeClassifier()
dt.fit(dense_train, y_train)
dt_yhat = dt.predict(dense_test)
dt_yhatprob = dt.predict_proba(dense_test)[:,1]
print("\nDecision Tree")
print("Accuracy = " + str(met.accuracy_score(y_test, dt_yhat)))
print("Recall = " + str(met.recall_score(y_test, dt_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, dt_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, dt_yhat)))
print("AUROC = " + str(met.roc_auc_score(y_test, dt_yhatprob)))


abdt = ens.AdaBoostClassifier()
abdt.fit(dense_train, y_train)
abdt_yhat = abdt.predict(dense_test)
abdt_yhatprob = abdt.predict_proba(dense_test)[:,1]
print("\nAda Boost")
print("Accuracy = " + str(met.accuracy_score(y_test, abdt_yhat)))
print("Recall = " + str(met.recall_score(y_test, abdt_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, abdt_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, abdt_yhat)))
print("AUROC = " + str(met.roc_auc_score(y_test, abdt_yhatprob)))


bdt = ens.BaggingClassifier()
bdt.fit(dense_train, y_train)
bdt_yhat = bdt.predict(dense_test)
bdt_yhatprob = bdt.predict_proba(dense_test)[:,1]
print("\nBagging")
print("Accuracy = " + str(met.accuracy_score(y_test, bdt_yhat)))
print("Recall = " + str(met.recall_score(y_test, bdt_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, bdt_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, bdt_yhat)))
print("AUROC = " + str(met.roc_auc_score(y_test, bdt_yhatprob)))


gbdt = ens.GradientBoostingClassifier()
gbdt.fit(dense_train, y_train)
gbdt_yhat = gbdt.predict(dense_test)
gbdt_yhatprob = gbdt.predict_proba(dense_test)[:,1]
print("\nGradient Boosted Decision Tree")
print("Accuracy = " + str(met.accuracy_score(y_test, gbdt_yhat)))
print("Recall = " + str(met.recall_score(y_test, gbdt_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, gbdt_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, gbdt_yhat)))
print("AUROC = " + str(met.roc_auc_score(y_test, gbdt_yhatprob)))


rf = ens.RandomForestClassifier()
rf.fit(dense_train, y_train)
rf_yhat = rf.predict(dense_test)
rf_yhatprob = rf.predict_proba(dense_test)[:,1]
print("\nRandom Forest")
print("Accuracy = " + str(met.accuracy_score(y_test, rf_yhat)))
print("Recall = " + str(met.recall_score(y_test, rf_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, rf_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, rf_yhat)))
print("AUROC = " + str(met.roc_auc_score(y_test, rf_yhatprob)))


#Naive Bayes Models
nb = nbc.MultinomialNB()
nb.fit(dense_train, y_train)
nb_yhat = nb.predict(dense_test)
nb_yhatprob = nb.predict_proba(dense_test)[:,1]
print("\nNaive Bayes")
print("Accuracy = " + str(met.accuracy_score(y_test, nb_yhat)))
print("Recall = " + str(met.recall_score(y_test, nb_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, nb_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, nb_yhat)))
print("AUROC = " + str(met.roc_auc_score(y_test, nb_yhatprob)))


#SVM Models
lsvm = svms.LinearSVC()
lsvm.fit(dense_train, y_train)
lsvm_yhat = lsvm.predict(dense_test)
print("\nLinear SVM")
print("Accuracy = " + str(met.accuracy_score(y_test, lsvm_yhat)))
print("Recall = " + str(met.recall_score(y_test, lsvm_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, lsvm_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, lsvm_yhat)))

#I can try out different kernel functions with this one
rbsvm = svms.SVC( kernel = "rbf")
rbsvm.fit(dense_train, y_train)
rbsvm_yhat = rbsvm.predict(dense_test)
print("\nRadial Basis SVM")
print("Accuracy = " + str(met.accuracy_score(y_test, rbsvm_yhat)))
print("Recall = " + str(met.recall_score(y_test, rbsvm_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, rbsvm_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, rbsvm_yhat)))

#SVM using a polynomial kernel
"""
ksvm = svms.SVC( kernel = "poly")
ksvm.fit(dense_train, y_train)
ksvm_yhat = ksvm.predict(dense_test)
print("\nPolynomial SVM")
print("Accuracy = " + str(met.accuracy_score(y_test, ksvm_yhat)))
print("Recall = " + str(met.recall_score(y_test, ksvm_yhat)) +
      "\nPrecision = " + str(met.precision_score(y_test, ksvm_yhat)) +
      "\nF-Statistic = " + str(met.f1_score(y_test, ksvm_yhat)))
"""

#Neural Net Models
ann = Sequential()

#adding layers to the model
ann.add(Dense(24, input_dim=24))
ann.add(Dense(4, activation='relu'))
ann.add(Dense(1, activation='sigmoid'))

#compiling the model
ann.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting the model
ann.fit(dense_train.as_matrix(), y_train.as_matrix(), epochs = 5, batch_size = 1)

#getting the accuracy for both the validation and train data
ann_accuracy = ann.evaluate(dense_test.as_matrix(), y_test.as_matrix())[1]

print("\n\nNeural Nets\nAccuracy = " + str(ann_accuracy))


#This is to visualize all of the different performances
accuracy_scores = [met.accuracy_score(y_test, lrl1_yhat), met.accuracy_score(y_test, lrl2_yhat),
                met.accuracy_score(y_test, knn_yhat), met.accuracy_score(y_test, dt_yhat), met.accuracy_score(y_test, abdt_yhat),
                met.accuracy_score(y_test, bdt_yhat), met.accuracy_score(y_test, gbdt_yhat), met.accuracy_score(y_test, rf_yhat),
                met.accuracy_score(y_test, nb_yhat), met.accuracy_score(y_test, sgdc_yhat), met.accuracy_score(y_test, lsvm_yhat),
                met.accuracy_score(y_test, rbsvm_yhat), ann_accuracy ]

recall_scores = [met.recall_score(y_test, lrl1_yhat), met.recall_score(y_test, lrl2_yhat),
                met.recall_score(y_test, knn_yhat), met.recall_score(y_test, dt_yhat), met.recall_score(y_test, abdt_yhat),
                met.recall_score(y_test, bdt_yhat), met.recall_score(y_test, gbdt_yhat), met.recall_score(y_test, rf_yhat),
                met.recall_score(y_test, nb_yhat), met.recall_score(y_test, sgdc_yhat), met.recall_score(y_test, lsvm_yhat),
                met.recall_score(y_test, rbsvm_yhat)]

precision_scores = [met.precision_score(y_test, lrl1_yhat), met.precision_score(y_test, lrl2_yhat),
                met.precision_score(y_test, knn_yhat), met.precision_score(y_test, dt_yhat), met.precision_score(y_test, abdt_yhat),
                met.precision_score(y_test, bdt_yhat), met.precision_score(y_test, gbdt_yhat), met.precision_score(y_test, rf_yhat),
                met.precision_score(y_test, nb_yhat), met.precision_score(y_test, sgdc_yhat), met.precision_score(y_test, lsvm_yhat),
                met.precision_score(y_test, rbsvm_yhat)]

f1_scores = [met.f1_score(y_test, lrl1_yhat), met.f1_score(y_test, lrl2_yhat), met.f1_score(y_test, knn_yhat),
                met.f1_score(y_test, dt_yhat), met.f1_score(y_test, abdt_yhat), met.f1_score(y_test, bdt_yhat),
                met.f1_score(y_test, gbdt_yhat),  met.f1_score(y_test, rf_yhat), met.f1_score(y_test, nb_yhat),
                met.f1_score(y_test, sgdc_yhat), met.f1_score(y_test, lsvm_yhat), met.f1_score(y_test, rbsvm_yhat) ]

roc_scores = [met.roc_auc_score(y_test, lrl1_yhatprob), met.roc_auc_score(y_test, lrl2_yhatprob), met.roc_auc_score(y_test, knn_yhatprob),
                met.roc_auc_score(y_test, dt_yhatprob), met.roc_auc_score(y_test, abdt_yhatprob), met.roc_auc_score(y_test, bdt_yhatprob),
                met.roc_auc_score(y_test, gbdt_yhatprob), met.roc_auc_score(y_test, rf_yhatprob), met.roc_auc_score(y_test, nb_yhatprob) ]

#This is the graph of all of the different scores
labels = ["LR l1", "LR l2", "KNN", "DT", "ABDT", "BDT", "GBDT", "RF", "NB", "SGD", "LSVM", "RBSVM", "ANN"]
legend = ["Accuracy", "Recall", "Precision" ,"F-Statistic", "ROC"]
x = list(range(13))
pp.rcParams["figure.figsize"] = [15,10]
pp.xticks(x, labels)
pp.plot(x, accuracy_scores)
pp.plot(recall_scores)
pp.plot(precision_scores)
pp.plot(f1_scores)
pp.plot(roc_scores)
pp.legend(legend)
pp.title("Basic Performance of Different Models on Dataset")
pp.xlabel("Models")
pp.axis()
pp.ylabel("Percentage")
#pp.savefig("models.jpg")
