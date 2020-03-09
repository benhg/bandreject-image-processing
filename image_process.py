import numpy as np
import numpy.fft as fp
import matplotlib.pyplot as plt
from pylab import *
 
PAD_WIDTH = 100

# Read in image, create padded image
im = imread('nasa.png')
im1 = np.zeros((im.shape[0]+2*PAD_WIDTH, im.shape[1]+2*PAD_WIDTH))
print(im.shape, im1.shape)
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        im1[i+PAD_WIDTH,j+PAD_WIDTH] = im[i,j]

# Linear ramp-down on top
for j in range(im.shape[1]):
    eventual_brightness = im [0][j]
    for i, val in enumerate(np.linspace(0, eventual_brightness, PAD_WIDTH)):
        im1 [i,j+PAD_WIDTH] = val
        eventual_brightness = im[j][i]
# Linear ramp-down on bottom
for j in range(im.shape[1]):
    eventual_brightness = im [im.shape[0]-1][j]
    for i, val in enumerate(np.linspace(eventual_brightness, 0, PAD_WIDTH)):
        im1 [i+im.shape[0]+ PAD_WIDTH, j+PAD_WIDTH] = val
        eventual_brightness = im[j][i]
# Linear ramp-down on left
for j in range(im.shape[0]):
    eventual_brightness = im [j][0]
    for i, val in enumerate(np.linspace(0, eventual_brightness, PAD_WIDTH)):
        im1 [j+PAD_WIDTH, i] = val
        eventual_brightness = im[j][i]
# Linear ramp-down on right
for j in range(im.shape[0]):
    eventual_brightness = im [j][im.shape[1]-1]
    for i, val in enumerate(np.linspace(eventual_brightness, 0, PAD_WIDTH)):
        im1 [j+PAD_WIDTH, i+im.shape[1]+PAD_WIDTH] = val
        eventual_brightness = im[j][i]



 
# the LPF kernel
kernel = [[1]*im1.shape[1]]*im1.shape[0]

plt.figure(figsize=(15,10))
plt.gray() # show the filtered result in grayscale

# Draw the ring in the objective PFP
freq = fp.fft2(im1)
freq_kernel = fp.fft2(fp.ifftshift(kernel))
for i, x in enumerate(freq_kernel):
    for j,y in enumerate(x):
        freq_kernel[i][j] = 1+0j
import math
# draw the circle
for angle in range(0, 3600, 1):
    angle = angle/10
    for r in range(1500,1700):
        r = r/10
        x = r * math.sin(math.radians(angle)) + 0
        y = r * math.cos(math.radians(angle)) + 0
        freq_kernel[int(round(y))][int(round(x))] = 0j
# Add a square in the circle:
for t in range(0, 25, 1):
    for j in range(-120, 100, 1):
        freq_kernel [t-120][j] = 0j
        freq_kernel [t+100][j] = 0j
for t in range(0, 25, 1):
    for j in range(-120, 100, 1):
        freq_kernel [j][t-120] = 0j
        freq_kernel [j][t+100] = 0j


# Monkey with the image by convolution
freq_LPF = freq*freq_kernel # by the Convolution theorem
im2 = fp.ifft2(freq_LPF)
freq_im2 = fp.fft2(im2)
 

# Create image
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