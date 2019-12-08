# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score as cvs #maybe try this in place of train_test_split()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

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

out_tensor=Flatten()(out_tensor)

#out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)

#out_tensor=tf.keras.layers.GlobalMaxPooling2D()(out_tensor)

out_tensor=Dense(130, activation = 'relu')(out_tensor)
#out_tensor=Dense(50, activation = 'relu')(out_tensor)
out_tensor=Dense(17, activation='softmax')(out_tensor)

# Define the full model by the endpoints.
model = tf.keras.models.Model(
        inputs = [in_tensor], outputs = [out_tensor])

#model.layers[-6].trainable=True
#model.layers[-12].trainable=True
#model.layers[].trainable=True


#feature extraction
#suggestions for further improvement: use better model; use multiple models then take class probabilities of each of them then take the average/median (or majority vote)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

X=np.load('pil1/data_pil1X.npy', allow_pickle=True)
y=np.load('pil1/data_pil1y.npy', allow_pickle=True)


#split the data - training, testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, 
        train_size = 0.8, 
        random_state=42)

y_train = to_categorical(y_train,17)
y_test = to_categorical(y_test,17)


#Classifiers
checkpoint = ModelCheckpoint("mobile_model.hdf5", monitor='loss', verbose=1,save_best_only=True,save_weights_only=True, mode='auto', period=1)
model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test),callbacks=[checkpoint])


# Testing
#feature extraction
y_train1=to_categorical(y)

checkpoint1 = ModelCheckpoint("mobile_model1.hdf5", monitor='loss', verbose=1,save_best_only=True,save_weights_only=True, mode='auto', period=1)
model.fit(X, y_train1, epochs = 10,callbacks=[checkpoint1])

Xt=np.load('pil1/data_pil1Xt.npy', allow_pickle=True)

pred=model.predict(Xt)

#creating submission file
q=[]
for i in pred:
    #q.append(class_names[i[]])
    q.append(class_names[np.where(i == np.amax(i))[0][0]])
    
index=0
with open("submission72.csv", "w") as fp:
    fp.write("Id,Category\n")
    for label in q:
        fp.write("%d,%s\n" % (index, label))
        index+=1



#kaggle accuracy= 0.80

