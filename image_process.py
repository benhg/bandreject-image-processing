import numpy as np
import numpy.fft as fp
import matplotlib.pyplot as plt
from pylab import *
 
im = imread('nasa.png')
 

xs = im.shape[0]
ys = im.shape[1]
# the LPF kernel

kernel = [[0]*xs]*ys
for i,x in enumerate(kernel):
    for j,y in enumerate(x):
        if abs((5*i)**2 + (3*j)**2 - 1**2) < 5**2:
            kernel[i][j] = 1000000
            print("found point")

 
plt.figure(figsize=(15,10))
plt.gray() # show the filtered result in grayscale
 
freq = fp.fft2(im)
freq_kernel = fp.fft2(fp.ifftshift(kernel))
#freq_LPF = freq*freq_kernel # by the Convolution theorem
#im2 = fp.ifft2(freq_LPF)
#freq_im2 = fp.fft2(im2)
 
plt.subplot(2,3,1)
plt.imshow(im)
plt.title('Original Image', size=20)
plt.subplot(2,3,2)
plt.imshow(im)
plt.title('Padded Image', size=20)
plt.subplot(2,3,3)
plt.imshow( (20*np.log10( 0.1 + fp.fftshift(freq))).astype(int), cmap='jet')
plt.title('Original Image Spectrum', size=20)
plt.subplot(2,3,4)
plt.imshow( kernel)#(20*np.log10( 0.1 + fp.fftshift(kernel))).astype(int), cmap='jet')
plt.title('Image Spectrum of the LPF', size=20)
plt.subplot(2,3,5)
#plt.imshow( (20*np.log10( 0.1 + fp.fftshift(freq_im2))).astype(int), cmap='jet')
plt.title('Image Spectrum after LPF', size=20)
plt.subplot(2,3,6)
#plt.imshow(im2.astype(np.uint8)) # the imaginary part is an artifact
plt.title('Output Image', size=20)
plt.show()