import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from simplelbp import local_binary_pattern
from math import floor, ceil

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score


def bilinear(image, r, c):
    minr = floor(r)
    minc = floor(c)
    maxr = ceil(r)
    maxc = ceil(c)

    dr = r-minr
    dc = c-minc

    top = (1-dc)*image[minr,minc] + dc*image[minr,maxc]
    bot = (1-dc)*image[maxr,minc] + dc*image[maxr,maxc]

    return (1-dr)*top+dr*bot

def local_binary_pattern(image, P=8, R=1):
    rr = - R * np.sin(2*np.pi*np.arange(P, dtype=np.double) / P)
    cc = R * np.cos(2*np.pi*np.arange(P, dtype=np.double) / P)
    rp = np.round(rr, 5)
    cp = np.round(cc, 5)
    
    rows = image.shape[0]
    cols = image.shape[1]

    output = np.zeros((rows, cols))

    for r in range(R,rows-R):
        for c in range(R,cols-R):
            lbp = 0
            for i in range(P):
                if bilinear(image, r+rp[i], c+cp[i]) - image[r,c] >= 0:
                    lbp += 1<<i
                            
            output[r,c] = lbp

    return output

def load_data(folder):
   
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
            
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def extract_lbp_features(X, P = 8, R = 5):

    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)

# Test our loader

X, y = load_data(".")
F = extract_lbp_features(X)
print("X shape: " + str(X.shape))
print("F shape: " + str(F.shape))

#Loop different classifiers
classifiers = [
    KNeighborsClassifier(5, metric= "euclidean"),
    LinearDiscriminantAnalysis(),
    SVC(kernel= 'rbf',C=1),
    LogisticRegression()]
names = ['KNN','LDA','SVC','LR']
for name, clf in zip(names,classifiers):
    scores = cross_val_score(clf, F, y, cv = 5)
    print(name," Accuracy: %0.2f" % (scores.mean()))

#Classify with enesemble classifiers
#Random Forest Classifier    
from sklearn.ensemble import RandomForestClassifier as RFC
clf = RFC(n_estimators=100, max_depth=2,random_state=0)
scores_2 = cross_val_score(clf, F, y, cv = 5)
print(" Accuracy Random Forest : %0.7f" % (scores_2.mean()))

#Extremely randomized classifier
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
scores_3 = cross_val_score(forest, F, y, cv = 5)
print(" Accuracy Extremely Randomized : %0.7f" % (scores_3.mean()))

#AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
scores_4 = cross_val_score(ada, F, y, cv = 5)
print(" Accuracy AdaBoost: %0.7f" % (scores_4.mean()))

#Gradient Boosted Tree classifier 
from sklearn import ensemble
gbb = ensemble.GradientBoostingClassifier()
scores_5 = cross_val_score(gbb, F, y, cv = 5)
print(" Accuracy Gradient Boosted: %0.7f" % (scores_5.mean()))