"""
Creates four SVM kernals for the given TV news station file 

@author Nicole Croutwater (nmcroutw)
"""

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
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



 # Linear
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

# Polynomial
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train, y_train)
print('\nPolynomial')
y_pred_poly = clf_poly.predict(X_test)
poly_accuracy = accuracy_score(y_test, y_pred_poly)
print('Accuracy: ',poly_accuracy)
poly_precision = precision_score(y_test, y_pred_poly)
print('Precision: ',poly_precision)
poly_recall = recall_score(y_test, y_pred_poly)
print('Recall: ',poly_recall)
poly_fmeasure = f1_score(y_test, y_pred_poly)
print('F-measure: ',poly_fmeasure)

# Radial Basic Function (Gaussian kernal)
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train, y_train)
print('\nRadial Basic Function')
#print(clf.score(X_test, y_test))
y_pred_rbf = clf_rbf.predict(X_test)
#print('fmeasure: ',f1_score(y_test, y_pred))
rbf_accuracy = accuracy_score(y_test, y_pred_rbf)
print('Accuracy: ',rbf_accuracy)
rbf_precision = precision_score(y_test, y_pred_rbf)
print('Precision: ',rbf_precision)
rbf_recall = recall_score(y_test, y_pred_rbf)
print('Recall: ', rbf_recall)
rbf_fmeasure = f1_score(y_test, y_pred_rbf)
print('F-measure: ',rbf_fmeasure)

# Sigmoid
clf_sig = svm.SVC(kernel='sigmoid')
clf_sig.fit(X_train, y_train)
print('\nSigmoid')
#print(clf.score(X_test, y_test))
y_pred_sigmoid = clf_sig.predict(X_test)
sigmoid_accuracy = accuracy_score(y_test, y_pred_sigmoid)
print('Accuracy: ', sigmoid_accuracy)
sigmoid_precision = precision_score(y_test, y_pred_sigmoid)
print('Precision: ',sigmoid_precision)
sigmoid_recall = recall_score(y_test, y_pred_sigmoid)
print('Recall: ', sigmoid_recall)
sigmoid_fmeasure = f1_score(y_test, y_pred_sigmoid)
print('F-measure: ', sigmoid_fmeasure)






"""
Labels : - +1/-1 ( Commercials/Non Commercials) 
Feature
Dimension Index in feature File
Shot Length
1
Motion Distribution( Mean and Variance)
2 - 3
Frame Difference Distribution ( Mean and Variance)
4 - 5
Short time energy ( Mean and Variance)
6 – 7 
ZCR( Mean and Variance)
8 - 9
Spectral Centroid ( Mean and Variance)
10 - 11
Spectral Roll off ( Mean and Variance)
12 - 13
Spectral Flux ( Mean and Variance)
14 - 15
Fundamental Frequency ( Mean and Variance)
16 - 17
Motion Distribution ( 40 bins)
18 -  58
Frame Difference Distribution ( 32 bins)
59 - 91
Text area distribution (  15 bins Mean  and 15 bins for variance )
92 - 122
Bag of Audio Words ( 4000 bins)
123 -  4123
Edge change Ratio ( Mean and Variance)
4124 - 4125
"""
