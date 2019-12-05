from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_selection import RFECV 
import matplotlib.pyplot as plt
from scipy.io import loadmat 
from sklearn.metrics import accuracy_score

mat = loadmat("arcene")
print(mat.keys()) 

#load dictonary data in arrays
p_test = mat["X_test"] 
p_train = mat["X_train"]
q_test = mat["y_test"].ravel()
q_train = mat["y_train"].ravel()
 

estimator = LR()
rfe = RFECV(estimator, step=50, verbose = 1)
rfe = rfe.fit(p_train, q_train)
rfe.support_

plt.plot(range(0,10001,50), rfe.grid_scores_)

pred = rfe.predict(p_test)
a = accuracy_score(q_test,pred)
print('Accuracy : ',a)

