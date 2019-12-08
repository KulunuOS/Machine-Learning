# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import os
import numpy as np
#from matplotlib.pyplot import plot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from PIL import Image 
from sklearn.metrics import accuracy_score

#remove warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

#from keras.applications.mobilenet import MobileNet
class_names = sorted(os.listdir(r"C:\Work\sgndataset\train"))
base_model = tf.keras.applications.mobilenet.MobileNet( 
        input_shape = (224,224,3), 
        include_top = False)

in_tensor = base_model.inputs[0]
out_tensor = base_model.outputs[0] 

# Add an average pooling layer (averaging each of the 1024 channels): 
out_tensor = tf.keras.layers.GlobalAveragePooling2D()(
        out_tensor)

# Define the full model by the endpoints.
model = tf.keras.models.Model(
        inputs = [in_tensor], outputs = [out_tensor])

model.compile(loss = "categorical_crossentropy",
              optimizer = 'sgd')


X = [] # Feature vectors will go here. 
y = [] 

for root, dirs, files in os.walk(r"C:\Work\sgndataset\train"): 
    for name in files: 
        if name.endswith(".jpg"):
            # Load the image: 
            img = Image.open(root + os.sep + name)
            # Resize it to the net input size: 
            img = img.resize((224,224))
            # Convert the data to float, and remove mean: 
#            img = img.astype(np.float32) 
#            img -= 128
            #img=img.convert(mode='F')
            img=np.array(img)
            img=img.astype(np.float32)
            img -= 128
    
            # Push the data through the model: 
            x = model.predict(img[np.newaxis, ...])[0]
            # And append the feature vector to our list. 
            X.append(x)
            # Extract class name from the directory name: 
            label = (root+os.sep+name).split(os.sep)[-2] 
            y.append(class_names.index(label))

X = np.array(X) 
y = np.array(y)

#split the data - training, testing
X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        train_size = 0.8, 
        random_state=42)
""""
classifiers = [
    LDA(),
    SVC(kernel= 'linear',C=10),
    SVC(kernel= 'rbf',C=10),
    LR(),
    RFC()]

names = ['LDA','SVC-Linear ','SVC-rbf','LR','Rand forest']

for name, clf in zip(names,classifiers):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    a = accuracy_score(y_test,pred)
    print('Accuracy of ',name,' ',a)
"""

#2
clf_LDA = LDA()
clf_LDA.fit(X_train, y_train)
pred = clf_LDA.predict(X_test)
a_LDA = accuracy_score(y_test,pred)
print('Accuracy of : ', a_LDA)

#4
clf_SVC_linear = SVC(kernel= 'linear',C=1)
clf_SVC_linear.fit(X_train, y_train)
pred = clf_SVC_linear.predict(X_test)
a_SVC_linear = accuracy_score(y_test,pred)
print('Accuracy of : ', a_SVC_linear)


#1
clf_SVC_rbf = SVC(kernel= 'rbf',C=1)
clf_SVC_rbf.fit(X_train, y_train)
pred = clf_SVC_rbf.predict(X_test)
a_SVC_rbf = accuracy_score(y_test,pred)
print('Accuracy of : ', a_SVC_rbf)

#3
clf_LR = LR()
clf_LR.fit(X_train, y_train)
pred = clf_LR.predict(X_test)
a_LR = accuracy_score(y_test,pred)
print('Accuracy of : ', a_LR)

#5
clf_RFC = RFC()
clf_RFC.fit(X_train, y_train)
pred = clf_RFC.predict(X_test)
a_RFC = accuracy_score(y_test,pred)
print('Accuracy of : ', a_RFC)


# Testing

Xt = [] # Feature vectors will go here. 
yt = []
for roott, dirst, filest in os.walk(r"C:\Work\sgndataset\testset"): 
    for namet in filest: 
        if namet.endswith(".jpg"):
            # Load the image: 
            imgt = Image.open(roott + os.sep + namet)
            # Resize it to the net input size: 
            imgt = imgt.resize((224,224))
            # Convert the data to float, and remove mean: 
#            img = img.astype(np.float32) 
#            img -= 128
            #img=img.convert(mode='F')
            imgt=np.array(imgt)
            imgt=imgt.astype(np.float32)
            imgt -= 128
    
            # Push the data through the model: 
            xt = model.predict(imgt[np.newaxis, ...])[0]
            # And append the feature vector to our list. 
            Xt.append(xt)
            # Extract class name from the directory name: 
#            labelt = (roott+os.sep+namet).split(os.sep)[-2] 
#            yt.append(class_names.index(labelt))

Xt = np.array(Xt) 
#yt = np.array(yt)

pred = clf_SVC_rbf.predict(Xt)
#test_accuracy = accuracy_score(yt,pred)
#print('Accuracy of : ', test_accuracy)
q=[]
for i in pred:
    q.append(class_names[i])
    
index=0
with open("submission1.csv", "w") as fp:
    fp.write("Id;Category\n")
    for label in q:
        fp.write("%d;%s\n" % (index, label))
        index+=1

        
