import cv2
import numpy as np
from matplotlib import pyplot as plt

#adding images
imgOrig = cv2.imread('ATU1.jpg',)

#greyscaling
gray_image = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

#Harris corners
dst = cv2.cornerHarris(gray_image, 2, 3, k=0.04)

#showing images
cv2.imshow('Original', imgOrig)
cv2.waitKey(0)
cv2.imshow('GrayScale', gray_image)
cv2.waitKey(0)
cv2.imshow('Harris corners', dst)
cv2.waitKey(0)
#plt.show()