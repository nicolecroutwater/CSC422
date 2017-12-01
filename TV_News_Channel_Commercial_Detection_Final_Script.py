# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:26:53 2017

@author: jtwall3

This is my final script in which I do my cross validation on the models that I selected as the ones that performed the best on the training
data. I included a lot of code that is commented out to show some of the stuff that I initially did but then decided against doing just
so there is proof that I did what I said I did.
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
import sklearn.datasets as ds
import sklearn.preprocessing as ppr
import scipy.sparse
import matplotlib.pyplot as pp

#This is so all of these models can be evaluated on each of the different tv news datsets
for j in ["BBC.txt", "CNN.txt", "CNNIBN.txt", "NDTV.txt", "TIMESNOW.txt"]:

    variables, truth = ds.load_svmlight_file("C:/Users/jtwall/Files/5Jr/CSC422/Project/TV_News_Channel_Commercial_Detection_Dataset/" + j, dtype = np.float32)

    pre_y = pd.DataFrame(truth)

    #This makes the true and false values go from 1 and -1 to 1 and 0
    y = []

    for index, row in pre_y.iterrows():
        if (row[0] == -1):
            y.append(0)
        else:
            y.append(1)

    y = pd.Series(y)

    #Getting the entire sparse dataframe
    sparse = pd.DataFrame(variables.toarray())

    #This is to drop missing data
    missingrows = []

    for index, row in sparse.iterrows():
        if (row[3] == 0):
            missingrows.append(index)

    sparse.drop(missingrows)

    """
    This is commented out because afer I saw the accuracies of the dense data and how much lower they were than the sparse data, I decided
    not to use my dense data

    #Getting the dense parts of the dataset
    dense = sparse[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,4123,4124]]

    dense.columns = ["Length", "MD_M", "MD_V", "FDD_M", "FDD_V", "STE_M", "STE_V", "ZCR_M", "ZCR_V",
                "SC_M", "SC_V", "SRO_M", "SRO_V", "SF_M", "SF_V", "FF_M", "FF_V", "ECR_M", "ECR_V"]

    #Separating out the different sparse parts of the matrix
    md = sparse[list(range(17,57))]
    fdd = sparse[list(range(57,89))]
    tad = sparse[list(range(89,119))]
    bow = sparse[list(range(119,4123))]

    #getting the number of nonzero values in each of the commercials
    dense["MD_N"] = md.astype(bool).sum(axis=1)
    dense["FDD_N"] = fdd.astype(bool).sum(axis=1)
    dense["TAD_N"] = tad.astype(bool).sum(axis=1)/2
    dense["TAD_S"] = tad.sum(axis=1)
    dense["BOW_N"] = bow.astype(bool).sum(axis=1)

    #Standardizing the data
    scaler = ppr.MinMaxScaler(feature_range=(0,1))
    dense = pd.DataFrame(scaler.fit_transform(dense))
    dense.columns = ["Length", "MD_M", "MD_V", "FDD_M", "FDD_V", "STE_M", "STE_V", "ZCR_M", "ZCR_V",
                "SC_M", "SC_V", "SRO_M", "SRO_V", "SF_M", "SF_V", "FF_M", "FF_V", "ECR_M", "ECR_V",
                "MD_N", "FDD_N", "TAD_N", "TAD_S", "BOW_N"]

    #This displays the distributions for each feature
    for i in dense.columns:
        pp.plot(range(dense.shape[0]), dense[i].sort_values())
        pp.title(i)
        pp.show()
    """

    #Cross Validation
    kf = ms.KFold(n_splits = 10, random_state = 72)

    """
    This is also commented out because these models were built on dense data and they did not perform well enough to make the
    final models.

    #Logistic Regression
    lrl1 = glm.LogisticRegression(penalty = "l1", solver = "liblinear")
    lrl1_scores = [0,0,0,0,0]

    for train_index, test_index in kf.split(dense):
        lrl1.fit(dense.iloc[train_index.tolist()], y.iloc[train_index.tolist()])
        lrl1_yhat = lrl1.predict(dense.iloc[test_index.tolist()])
        lrl1_yhatprob = lrl1.predict_proba(dense.iloc[test_index.tolist()])[:,1]

        lrl1_scores[0] += met.accuracy_score(y[test_index.tolist()], lrl1_yhat)*.1
        lrl1_scores[1] += met.recall_score(y[test_index.tolist()], lrl1_yhat)*.1
        lrl1_scores[2] += met.precision_score(y[test_index.tolist()], lrl1_yhat)*.1
        lrl1_scores[3] += met.f1_score(y[test_index.tolist()], lrl1_yhat)*.1
        lrl1_scores[4] += met.roc_auc_score(y[test_index.tolist()], lrl1_yhatprob)*.1

    print(lrl1_scores)

    #Gradient Boosted Decision Tree
    gbdt = ens.GradientBoostingClassifier()
    gbdt_scores = [0,0,0,0,0]

    for train_index, test_index in kf.split(dense):
        gbdt.fit(dense.iloc[train_index.tolist()], y.iloc[train_index.tolist()])
        gbdt_yhat = gbdt.predict(dense.iloc[test_index.tolist()])
        gbdt_yhatprob = gbdt.predict_proba(dense.iloc[test_index.tolist()])[:,1]

        gbdt_scores[0] += met.accuracy_score(y[test_index.tolist()], gbdt_yhat)*.1
        gbdt_scores[1] += met.recall_score(y[test_index.tolist()], gbdt_yhat)*.1
        gbdt_scores[2] += met.precision_score(y[test_index.tolist()], gbdt_yhat)*.1
        gbdt_scores[3] += met.f1_score(y[test_index.tolist()], gbdt_yhat)*.1
        gbdt_scores[4] += met.roc_auc_score(y[test_index.tolist()], gbdt_yhatprob)*.1

    print(gbdt_scores)

    #Random Forest
    rf = ens.RandomForestClassifier(n_estimators=50)
    rf_scores = [0,0,0,0,0]

    for train_index, test_index in kf.split(dense):
        rf.fit(dense.iloc[train_index.tolist()], y.iloc[train_index.tolist()])
        rf_yhat = rf.predict(dense.iloc[test_index.tolist()])
        rf_yhatprob = rf.predict_proba(dense.iloc[test_index.tolist()])[:,1]

        rf_scores[0] += met.accuracy_score(y[test_index.tolist()], rf_yhat)*.1
        rf_scores[1] += met.recall_score(y[test_index.tolist()], rf_yhat)*.1
        rf_scores[2] += met.precision_score(y[test_index.tolist()], rf_yhat)*.1
        rf_scores[3] += met.f1_score(y[test_index.tolist()], rf_yhat)*.1
        rf_scores[4] += met.roc_auc_score(y[test_index.tolist()], rf_yhatprob)*.1

    print(rf_scores)
    """

    #In regards to hyper paramter tuning, I tuned a few of the varibales manually to explore but ended up that everything that I changed
    #caused my testing metrics to go down significantly so I did not code a GridSearch because I deemed it unecessary

    print(j)

    #Gradient Boosted Decision Tree on the sparse data
    gbdt = ens.GradientBoostingClassifier(random_state = 72)
    gbdt_sparse_scores = [0,0,0,0,0]

    for train_index, test_index in kf.split(sparse):
        gbdt.fit(sparse.iloc[train_index.tolist()], y.iloc[train_index.tolist()])
        gbdt_yhat = gbdt.predict(sparse.iloc[test_index.tolist()])
        gbdt_yhatprob = gbdt.predict_proba(sparse.iloc[test_index.tolist()])[:,1]

        gbdt_sparse_scores[0] += met.accuracy_score(y[test_index.tolist()], gbdt_yhat)*.1
        gbdt_sparse_scores[1] += met.recall_score(y[test_index.tolist()], gbdt_yhat)*.1
        gbdt_sparse_scores[2] += met.precision_score(y[test_index.tolist()], gbdt_yhat)*.1
        gbdt_sparse_scores[3] += met.f1_score(y[test_index.tolist()], gbdt_yhat)*.1
        gbdt_sparse_scores[4] += met.roc_auc_score(y[test_index.tolist()], gbdt_yhatprob)*.1

    print("GBDT")
    print(gbdt_sparse_scores)

    #Random Forest Classifier on the sparse data
    rf = ens.RandomForestClassifier(n_estimators=50, random_state = 72)
    rf_sparse_scores = [0,0,0,0,0]

    for train_index, test_index in kf.split(sparse):
        rf.fit(sparse.iloc[train_index.tolist()], y.iloc[train_index.tolist()])
        rf_yhat = rf.predict(sparse.iloc[test_index.tolist()])
        rf_yhatprob = rf.predict_proba(sparse.iloc[test_index.tolist()])[:,1]

        rf_sparse_scores[0] += met.accuracy_score(y[test_index.tolist()], rf_yhat)*.1
        rf_sparse_scores[1] += met.recall_score(y[test_index.tolist()], rf_yhat)*.1
        rf_sparse_scores[2] += met.precision_score(y[test_index.tolist()], rf_yhat)*.1
        rf_sparse_scores[3] += met.f1_score(y[test_index.tolist()], rf_yhat)*.1
        rf_sparse_scores[4] += met.roc_auc_score(y[test_index.tolist()], rf_yhatprob)*.1

    print("RF")
    print(rf_sparse_scores)
