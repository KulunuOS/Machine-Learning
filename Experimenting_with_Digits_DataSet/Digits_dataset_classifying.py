from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

#load digits
digits = load_digits()
print(digits.keys())

#plot the first image
plt.gray()
plt.imshow(digits.images[0])
plt.show()
print(digits.target[0])

#split the data - training, testing
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, train_size = 0.8, random_state=42)
print('   shape of X training set : ', np.shape(X_train))
print('   shape of X testing set  : ', np.shape(X_test))
print('   shape of X training set : ', np.shape(y_train))
print('   shape of X testing set  : ', np.shape(y_test))

classifiers = [
    KNeighborsClassifier(5, metric= "euclidean"),
    LinearDiscriminantAnalysis(),
    SVC(kernel= 'rbf',C=10),
    LogisticRegression()]

names = ['KNN','LDA','SVC','LR']

for name, clf in zip(names,classifiers):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    a = accuracy_score(y_test,pred)
    print('Accuracy of ',name,' ',a)
