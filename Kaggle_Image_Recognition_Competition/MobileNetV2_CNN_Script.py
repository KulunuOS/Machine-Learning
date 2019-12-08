#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score as cvs 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

#remove warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


# In[ ]:


#from keras.applications.mobilenet import MobileNet
class_names = sorted(os.listdir(r"C:\Work\sgndataset\train"))
base_model = tf.keras.applications.mobilenet.MobileNet( 
        input_shape = (224,224,3), 
        include_top = False)

in_tensor = base_model.inputs[0]
out_tensor = base_model.outputs[0] 

out_tensor=Flatten()(out_tensor)

out_tensor=Dense(130, activation = 'relu')(out_tensor)
out_tensor=Dense(17, activation='softmax')(out_tensor)

# Define the full model by the endpoints.
model = tf.keras.models.Model(
        inputs = [in_tensor], outputs = [out_tensor])

# In[ ]:

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

# In[ ]:
#Load data (from previously compiled data)

X=np.load('pil1/data_pil1X.npy', allow_pickle=True)
y=np.load('pil1/data_pil1y.npy', allow_pickle=True)
Xt=np.load('pil1/data_pil1Xt.npy', allow_pickle=True)


# In[]:

#split the data - training, testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, 
        train_size = 0.8, 
        random_state=42)

y_train = to_categorical(y_train,17)
y_test = to_categorical(y_test,17)



# In[ ]:
#Train 
checkpoint = ModelCheckpoint("mobile_model.hdf5", monitor='loss', verbose=1,save_best_only=True,save_weights_only=True, mode='auto', period=1)
#model.fit(X_train, y_train, epochs = 15, validation_data = (X_test, y_test),callbacks=[checkpoint])
model.fit(X_train, y_train,
          batch_size= 128,
          epochs=12,
          verbose=1,
          validation_data=(X_test, y_test))


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# In[ ]:
#predict test data labels
pred=model.predict(Xt)

# In[ ]:
#creating submission file
q=[]
for i in pred:
    #q.append(class_names[i[]])
    q.append(class_names[np.where(i == np.amax(i))[0][0]])
    
index=0
with open("submission6.csv", "w") as fp:
    fp.write("Id,Category\n")
    for label in q:
        fp.write("%d,%s\n" % (index, label))
        index+=1


