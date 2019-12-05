from sklearn.linear_model import LogisticRegression as LR
import numpy as np
from scipy.io import loadmat 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

mat = loadmat("arcene")
print(mat.keys()) 

#load dictonary data in arrays
p_test = mat["X_test"] 
p_train = mat["X_train"]
q_test = mat["y_test"].ravel()
q_train = mat["y_train"].ravel()

C_range = 10.0**np.arange(-4, 3)
clf = LR()

accuracy_mat = []
weights=[]
for C in C_range:
    clf.C = C
    clf.fit(p_train, q_train)
    pred = clf.predict(p_test)
    accuracy = 100.0*np.mean(pred == q_test)
    accuracy_mat.append(accuracy)
    weights.append(clf.coef_)
indices = np.argsort(accuracy_mat)[::-1]    
print("Accuracy for C = %.2e is %.1f %% (||w|| = %.4f)" % \
      (C_range[indices[:10]], 
       accuracy[indices[:10]], 
       np.linalg.norm(weights[indices[:10]])))

#plot sparseness
plt.figure()
plt.title("Feature importances")
#plt.hist(importances,bins=10)
plt.bar(range(100),accuracy_mat[indices[:100]])
plt.xticks(range(100), C_range[indices[:100]], rotation='vertical')
plt.show()


C_index = np.argmax(accuracy_mat)
clf.C = C_range[C_index]
clf.fit(p_train, q_train)
features = clf.coef_
print("Number of features : " ,len(features))

pred = clf.predict(p_test)
a = accuracy_score(q_test,pred)
print('Accuracy : ',a)