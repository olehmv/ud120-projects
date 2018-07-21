#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################

print "number of featers", len(features_train[0])

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
### fit the classifier on the training features and labels
t0 = time()
clf.fit(features_train, labels_train)### training time: 54.22 s
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)### predict time: 0.031 s
print "predict time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)
print accuracy ### 0.9766780432309442



#########################################################


