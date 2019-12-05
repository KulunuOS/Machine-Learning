import matplotlib.pyplot as plt
#import numpy as np
from scipy.io import loadmat 

mat = loadmat("twoClassData")
print(mat.keys()) 

#load dictonary data in arrays
X = mat["X"] 
y = mat["y"].ravel() #convert y to 1-D array

p = X[y == 0, :] #filter X values correspond to y==0
q = X[y == 1, :] #filter X values correspond to y==1

plt.plot(p[:,0], p[:, 1],'ro',q[:,0], q[:, 1],'bo')
plt.show()