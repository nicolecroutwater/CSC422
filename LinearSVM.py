#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates LinearSVM for a TV station given the file

@author: Nicole Croutwater (nmcroutw)
"""

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.datasets import load_svmlight_file


# read in data
data_X, data_y = load_svmlight_file("/Users/nicolecroutwater/Desktop/CSC 422/TV_News_Channel_Commercial_Detection_Dataset (1)/BBC.txt")

xFeatures = pd.DataFrame(data_X.toarray())
yTarget = pd.DataFrame(data_y)

# Split data into 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(xFeatures, yTarget,
                                                    stratify=yTarget, 
                                                    test_size=0.3, 
                                                    random_state = 72)


clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train, y_train)
print('Linear')
y_pred = clf_linear.predict(X_test)
linear_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ',linear_accuracy)
linear_precision = precision_score(y_test, y_pred)
print('Precision: ',linear_precision)
linear_recall = recall_score(y_test, y_pred)
print('Recall: ',linear_recall)
linear_fmeasure = f1_score(y_test, y_pred)
print('F-measure: ',linear_fmeasure)


