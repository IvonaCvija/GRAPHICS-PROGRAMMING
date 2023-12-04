import cv2
import numpy as np
from matplotlib import pyplot as plt

imgOrig = cv2.imread('ATU.jpg',)
cv2.imshow('Original', imgOrig)
#grayscaling an image
gray_image = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Grayscale', gray_image)

nrows = 2
ncols = 1

plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(imgOrig, 
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.show()