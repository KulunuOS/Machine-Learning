import os
import numpy as np
from PIL import Image

class_names = sorted(os.listdir(r"C:\Work\sgndataset\train"))
X = [] # Feature vectors will go here. 
y = [] 

for root, dirs, files in os.walk(r"C:\Work\sgndataset\train"): 
    for name in files: 
        if name.endswith(".jpg"):
            # Load the image: 
            img = Image.open(root + os.sep + name)
            # Resize it to the net input size: 
            img = img.resize((224,224)) #resizing can be done to higher dimensions to see if improving resolution improves classification accuracy
            #other stuff that can improve classification: resize with a different function (e.g. cv2); resize the larger dimension of the image to 224 (for example), keeping the aspect ratio, then padd the smaller dimension with zeros till it's at 224 too
            
            #Convert the data to float, and remove mean:
            img=np.array(img)
            img=img.astype(np.float32)
            img1=img
            img -= 128
            img2=img
#            img=img[np.newaxis, ...]
            X.append(img)
            # Push the data through the model: 
#            x = model.predict(img[np.newaxis, ...])[0]
#            # And append the feature vector to our list. 
#            X.append(x)
            # Extract class name from the directory name: 
            label = (root+os.sep+name).split(os.sep)[-2] 
            y.append(class_names.index(label))#in case of NN, convert to_categorical()

X = np.array(X) 
y = np.array(y) 


np.save('Imgset_1/data_X',X,allow_pickle=True)
np.save('Imgset_1/data_y',y,allow_pickle=True)

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
            imgt=np.array(imgt)
            imgt=imgt.astype(np.float32)
            imgt -= 128
            # Push the data through the model: 
            #xt = model.predict(imgt[np.newaxis, ...])[0]
            # And append the feature vector to our list. 
            Xt.append(imgt)

Xt = np.array(Xt)

np.save('Imgset_1/data_Xt',Xt,allow_pickle=True)