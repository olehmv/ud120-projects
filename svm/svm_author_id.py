#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

### slice the training dataset down to 1% of its original size, tossing out 99% of the training data
### training time: 0.078 s
### predict time: 0.852 s
### accuracy 0.8845278725824801
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

from sklearn import svm

# training time: 0.078 s
# predict time: 0.954 s
# accuracy 0.6160409556313993
clf = svm.SVC(kernel='rbf')

# The C parameter trades off misclassification of training examples against simplicity of the decision surface
# on 1% of full data set :
# training time: 0.071 s
# predict time: 0.782 s
# accuracy 0.8924914675767918
# on full data set :
# training time: 92.15 s
# predict time: 9.398 s
# accuracy 0.9908987485779295
clf = svm.SVC(kernel='rbf',C=10000.0)

# clf = svm.SVC(kernel='linear')

t0 = time()
clf.fit(features_train, labels_train)### training time: 140.479 s kernel='linear'
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)### predict time: 14.524 s kernel='linear'
print "predict time:", round(time()-t0, 3), "s"

answer=pred[10]
print(answer)

answer=pred[26]
print(answer)

answer=pred[50]
print(answer)


res=[a for a in pred if a==1]
print(len(res))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)
print accuracy ### 0.9840728100113766 kernel='linear'

#########################################################


