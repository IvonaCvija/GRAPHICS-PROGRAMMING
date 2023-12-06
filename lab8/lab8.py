import cv2
import numpy as np
from matplotlib import pyplot as plt

#original image
#imgOrig = cv2.imread('ATU.jpg',)
imgOrig = cv2.imread('st_martin_today.jpg',)
#cv2.imshow('Original', imgOrig)

#grayscaling an image
gray_image = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Grayscale', gray_image)

#(KernelSizeWidth, KernelSizeHeight) = 3
#blurred image
img3x3Blur = cv2.GaussianBlur(gray_image,(3, 3),0) #3x3

img13x13Blur = cv2.GaussianBlur(gray_image,(13, 13),0) #13x13

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

#showing plot 1
plt.show()

#Sobel - edge detection
sobelHorizontal = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5) # x dir
sobelVertical = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5) # y dir
sobelBoth = cv2.addWeighted(sobelHorizontal,1,sobelVertical,1,0)

#Canny - edge detection
canny = cv2.Canny(gray_image,165,380)

#thresholding
# If f (x, y) < T 
#    then f (x, y) = 0 
# else 
#    f (x, y) = 255

# where 
# f (x, y) = Coordinate Pixel Value
# T = Threshold Value.
#cv2.threshold(source, thresholdValue, maxVal, thresholdingTechnique) 
# all pixels value above 120 will be set to 255 
ret, imgWithThreshold = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY) 
ret, imgWithThreshold2 = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV) #Binary inverted
ret, imgWithThreshold3 = cv2.threshold(gray_image, 120, 255, cv2.THRESH_TRUNC) #Truncate 

nrows2 = 3
ncols2 = 3

plt.subplot(nrows2, ncols2,1),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Sobel Horizontal'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows2, ncols2,2),plt.imshow(sobelVertical, cmap = 'gray')
plt.title('Sobel Vertical'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows2, ncols2,3),plt.imshow(sobelBoth, cmap = 'gray')
plt.title('Sobel Horizontal and Vertical'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows2, ncols2,4),plt.imshow(canny, cmap = 'gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows2, ncols2,5),plt.imshow(imgWithThreshold, cmap = 'gray')
plt.title('Image With Threshold '), plt.xticks([]), plt.yticks([])
plt.subplot(nrows2, ncols2,6),plt.imshow(imgWithThreshold2, cmap = 'gray')
plt.title('Image With Threshold2 '), plt.xticks([]), plt.yticks([])
plt.subplot(nrows2, ncols2,7),plt.imshow(imgWithThreshold3, cmap = 'gray')
plt.title('Image With Threshold3 '), plt.xticks([]), plt.yticks([])


#showing plot 2
plt.show()

