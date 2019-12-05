import numpy as np
import matplotlib.pyplot as plt

x = np.zeros((800))
for n in range(400,500):
    x[n] = np.cos(2 * np.pi * 0.1 * n)
    
x_n = x + np.sqrt(0.5) * np.random.randn(x.size)

d = np.zeros((800))

d[0:100]= np.correlate(x[400:500],x_n[0:100], 'same')
d[100:200]= np.correlate(x[400:500],x_n[100:200], 'same')
d[200:300]= np.correlate(x[400:500],x_n[200:300], 'same')
d[300:400]= np.correlate(x[400:500],x_n[300:400], 'same')
d[400:500]= np.correlate(x[400:500],x_n[400:500], 'same')
d[500:600]= np.correlate(x[400:500],x_n[500:600], 'same')
d[600:700]= np.correlate(x[400:500],x_n[600:700], 'same')
d[700:800]= np.correlate(x[400:500],x_n[700:800], 'same')

h = np.zeros((800))
for n in range(400,500):
    h[n] = np.exp(-2*np.pi*1j*0.1*n)

y = np.zeros((800))
y[0:100] = np.abs(np.convolve(h[400:500], x_n[0:100], 'same'))
y[100:200] = np.abs(np.convolve(h[400:500], x_n[0:100], 'same'))
y[200:300] = np.abs(np.convolve(h[400:500], x_n[200:300], 'same'))
y[300:400] = np.abs(np.convolve(h[400:500], x_n[300:400], 'same'))
y[400:500] = np.abs(np.convolve(h[400:500], x_n[400:500], 'same'))            
y[500:600] = np.abs(np.convolve(h[400:500], x_n[500:600], 'same'))
y[600:700] = np.abs(np.convolve(h[400:500], x_n[600:700], 'same'))
y[700:800] = np.abs(np.convolve(h[400:500], x_n[700:800], 'same'))


n = np.arange(100)
h = np.exp(-2*np.pi*1j*0.1*n)
y = np.abs(np.convolve(h,x_n, 'same'))


fig, ax = plt.subplots(4, 1) 
ax[0].plot(x) 
ax[1].plot(x_n)
ax[2].plot(d)
ax[3].plot(y)
plt.show()