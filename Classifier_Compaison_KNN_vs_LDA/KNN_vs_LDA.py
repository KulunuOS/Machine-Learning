from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from scipy.io import loadmat 
mat = loadmat("twoClassData.mat")
print(mat.keys())

X = mat["X"]
y = mat["y"].ravel()

#p = X[y == 0, :]
#q = X[y == 1, :]
#plt.plot(p[:,0 ],p[:, 1],'ro',q[:,0 ],q[:, 1],'bo')
'''
te_data = X[:200]
tr_data = X[100:300]
te_labels = y[:200]
tr_labels= y[100:300]
'''
tr_data, te_data, tr_labels, te_labels = train_test_split(X, y, test_size=0.5, random_state=42)


#Testing KNN
model = KNeighborsClassifier(n_neighbors = 5, metric = "euclidean")
model.fit(tr_data, tr_labels)
pred_1 = model.predict(te_data)
print('Accuracy KNN % = ',accuracy_score(te_labels,pred_1))
#p = te_data[pred_1 == 0, :]
#q = te_data[pred_1 == 1, :]
#plt.plot(p[:,0 ],p[:, 1],'ro',q[:,0 ],q[:, 1],'bo')

#Testing LDA
clf = LinearDiscriminantAnalysis()
clf.fit(tr_data, tr_labels)
pred_2 = clf.predict(te_data)
#r = te_data[pred_2 == 0, :]
#s = te_data[pred_2 == 1, :]
#plt.plot(r[:,0 ],r[:, 1],'ro',s[:,0 ],s[:, 1],'bo')
print('Accuracy LDA % = ',accuracy_score(te_labels,pred_2))
