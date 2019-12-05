import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Read the data
img = imread("uneven_illumination.jpg") #Read an image from a file into an array.
plt.imshow(img, cmap='gray')
plt.title("Image shape is %dx%d" % (img.shape[1], img.shape[0]))
plt.show()

# Create the X-Y coordinate pairs in a matrix
X, Y = np.meshgrid(range(1300), range(1030))
Z = img

x = X.ravel()
y = Y.ravel()
z = Z.ravel()

# Create data matrix
# Use function "np.column_stack".
# Function "np.ones_like" creates a vector like the input.
H = np.column_stack([x**2,y**2,x*y,x,y,np.ones_like(x)])


# Use np.linalg.lstsq to solve coefficients
# Put coefficients to variable "theta" which we use below.
theta = np.linalg.lstsq(H,z)[0]

 
# Predict
z_pred = H @ theta #@ takes inner product in numpy
Z_pred = np.reshape(z_pred, X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z_pred,rstride=1, cstride=1, alpha=0.2)
plt.show()

# Subtract & show
S = Z - Z_pred
plt.imshow(S, cmap = 'gray')
plt.show()
