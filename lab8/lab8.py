import cv2
import numpy as np
from matplotlib import pyplot as plt

#original image
imgOrig = cv2.imread('ATU.jpg',)
cv2.imshow('Original', imgOrig)

#grayscaling an image
gray_image = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Grayscale', gray_image)

#(KernelSizeWidth, KernelSizeHeight) = 3
#blurred image
img3x3Blur = cv2.GaussianBlur(gray_image,(3, 3),0)

img13x13Blur = cv2.GaussianBlur(gray_image,(13, 13),0)



#num of rows and colums in a plot
nrows = 2
ncols = 2
#adding plots
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(imgOrig, 
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(img3x3Blur, cmap = 'gray')
plt.title('3x3 Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(img13x13Blur, cmap = 'gray')
plt.title('13x13 Blur'), plt.xticks([]), plt.yticks([])


#showing plots
plt.show()