import numpy as np
import numpy.fft as fp
import matplotlib.pyplot as plt
from pylab import *
 
im = imread('nasa.png')
im1 = np.zeros((im.shape[0], im.shape[1]))
print(im.shape, im1.shape)
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        im1[i,j] = im[i,j]
 
def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector
 
# the LPF kernel
kernel = [[1]*im.shape[1]]*im.shape[0]

plt.figure(figsize=(15,10))
plt.gray() # show the filtered result in grayscale

freq = fp.fft2(im1)
freq_kernel = fp.fft2(fp.ifftshift(kernel))
for i, x in enumerate(freq_kernel):
    for j,y in enumerate(x):
        freq_kernel[i][j] = 1+0j
import math
# draw the circle
for angle in range(0, 3600, 1):
    angle = angle/10
    for r in range(1100,1300):
        r = r/10
        x = r * math.sin(math.radians(angle)) + 0
        y = r * math.cos(math.radians(angle)) + 0
        freq_kernel[int(round(y))][int(round(x))] = 0j



freq_LPF = freq*freq_kernel # by the Convolution theorem
im2 = fp.ifft2(freq_LPF)
freq_im2 = fp.fft2(im2)
 
plt.subplot(2,3,1)
plt.imshow(im)
plt.title('Original Image', size=20)
plt.subplot(2,3,2)
plt.imshow(im1)
plt.title('Padded Image', size=20)
plt.subplot(2,3,3)
plt.imshow( (20*np.log10( 0.1 + fp.fftshift(freq))).astype(int), cmap='jet')
plt.title('Original Image Spectrum', size=20)
plt.subplot(2,3,4)
plt.imshow( (20*np.log10( 0.1 + fp.fftshift(freq_kernel))).astype(int), cmap='jet')
plt.title('Image Spectrum of the Filter', size=20)
plt.subplot(2,3,5)
plt.imshow( (20*np.log10( 0.1 + fp.fftshift(freq_im2))).astype(int), cmap='jet')
plt.title('Image Spectrum after Filtering', size=20)
plt.subplot(2,3,6)
plt.imshow(im2.astype(np.float64)) # the imaginary part is an artifact
plt.title('Output Image', size=20)
plt.show()