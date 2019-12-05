# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:37:03 2019

@author: - <

"""
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat 

mat = loadmat("arcene")
print(mat.keys()) 

#load dictonary data in arrays
p_test = mat["X_test"] 
p_train = mat["X_train"]
q_test = mat["y_test"].ravel()
q_train = mat["y_train"].ravel()
 

clf = RandomForestClassifier(n_estimators=100)
clf.fit(p_train,q_train)


P = p_train
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(P.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
#plt.hist(importances,bins=10)
plt.bar(range(10),importances[indices[:10]])
plt.xticks(range(10), importances[indices[:10]], rotation='vertical')
plt.show()


