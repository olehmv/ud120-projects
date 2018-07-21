#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################
from time import time


# training time: 0.062 s
# predict time: 0.016 s
# accuracy 0.92
from sklearn.ensemble import AdaBoostClassifier
t0 = time()
clf = AdaBoostClassifier(n_estimators=30,learning_rate=2)
clf.fit(features_train, labels_train)###
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)
print "accuracy", accuracy


# training time: 0.031 s
# predict time: 0.0 s
# accuracy 0.94, 0.904, 0.92, 0.928, 0.924
from sklearn.ensemble import RandomForestClassifier
t0 = time()
clf = RandomForestClassifier(max_depth=2)
clf.fit(features_train, labels_train)###
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)
print "accuracy", accuracy



# training time: 0.031 s
# predict time: 0.0 s
# accuracy 0.7
from sklearn.cluster import KMeans
t0 = time()
clf = KMeans(n_clusters=2, precompute_distances=True).fit(features_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)
print "accuracy", accuracy
print "labels", clf.labels_
print "cluster centers", clf.cluster_centers_


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary








try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
