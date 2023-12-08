import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

#adding images
imgOrig = cv2.imread('ATU1.jpg',)

#greyscaling
gray_image = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

#Harris corners
dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)

#deep copy
imgHarris = copy.deepcopy(imgOrig)

threshold = 0.35; #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
            if dst[i][j] > (threshold*dst.max()):
                    cv2.circle(imgHarris,(j,i),3,(120, 70, 185),-1)

#showing images
# cv2.imshow('Original', imgOrig)
# cv2.waitKey(0)
# cv2.imshow('GrayScale', gray_image)
# cv2.waitKey(0)
cv2.imshow('Harris corners', imgHarris)
cv2.waitKey(0)

#rows and colums for the plot
nrows = 2
ncols = 2
#plot
plt.subplot(nrows, ncols,1),plt.imshow(imgOrig, cmap = 'gray')
plt.title(' Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title(' Gray'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(imgHarris, cmap = 'gray')
plt.title(' Harris'), plt.xticks([]), plt.yticks([])


plt.show()