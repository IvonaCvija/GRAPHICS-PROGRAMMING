import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

#adding images
imgOrig1 = cv2.imread('ATU1.jpg',) #orb
imgOrig2 = cv2.imread('ATU2.jpg',)

imgOriginal = cv2.imread('malta.jpg',) #harris, shi tomasi 

#imgCD = cv2.imread('cat.jpg',) #contour detection
imgCD = cv2.imread('shapes.jpg',)

img = cv2.imread('temple.jpg',) #rgb, hsv image

ORBImgOrig1 = cv2.imread('coke.jpg',)
ORBImgOrig2 = cv2.imread('cokepolarbear.jpg',)

#greyscaling
gray_image = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
ORBgray_image1 = cv2.cvtColor(ORBImgOrig1, cv2.COLOR_BGR2GRAY) #coke
ORBgray_image2 = cv2.cvtColor(ORBImgOrig2, cv2.COLOR_BGR2GRAY)  #polar bear with coke
gray_imageCD = cv2.cvtColor(imgCD, cv2.COLOR_BGR2GRAY)

#Harris corners
dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)

#deep copy for Harris
imgHarris = copy.deepcopy(imgOriginal)

threshold = 0.15; #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
            if dst[i][j] > (threshold*dst.max()):
                    cv2.circle(imgHarris,(j,i),3,(120, 70, 185),-1)

#Shi Tomasi
corners = cv2.goodFeaturesToTrack(gray_image,88,0.01,10)

#deep copy for Shi Tomasi
imgShiTomasi = copy.deepcopy(imgOriginal)

for i in corners:
    x, y = i[0]  
    cv2.circle(imgShiTomasi, (int(x), int(y)), 3, (20, 252, 197), -1)

#ORB
# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(imgOriginal,None)

# compute the descriptors with ORB
kp, des = orb.compute(imgOriginal, kp)

# draw only keypoints location,not size and orientation
imgORB = cv2.drawKeypoints(imgOriginal, kp, None, color=(0,255,0), flags=0)

#ORB BRUTE-FORCE
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(ORBgray_image1,None)
kp2, des2 = orb.detectAndCompute(ORBgray_image2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
imgORBBruteForce = cv2.drawMatches(ORBgray_image1,kp1,ORBgray_image2,kp2,matches[:15],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#FLANN
# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgOrig1,None)
kp2, des2 = sift.detectAndCompute(imgOrig2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2
                .DrawMatchesFlags_DEFAULT)
imgORBFlann = cv2.drawMatchesKnn(imgOrig1,kp1,imgOrig2,kp2,matches,None,**draw_params)

#RGB SPLIT
#https://machinelearningknowledge.ai/split-and-merge-image-color-space-channels-in-opencv-and-numpy/
#Split Image with cv2.split
blue,green,red = cv2.split(img)

#HSV colorspace has three channels – Hue, Saturation, and value
#creating HSV image
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Split the HSV image into its components
h_img, s_img, v_img = cv2.split(imgHSV)

#Contour Detection
#https://medium.com/mlearning-ai/how-to-detect-contours-in-an-image-in-python-using-opencv-3c245cb1d4d
#thresholding
ret, thresh = cv2.threshold(gray_imageCD, 125, 255, 0)

#detect contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

copy_imgCD = imgCD.copy()
cv2.drawContours(copy_imgCD,contours,-1,(0,0,255),2)


#PLOT
#rows and colums for the plot
nrows = 2
ncols = 3
#plot
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(imgOriginal, 
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title(' Original '), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(gray_image, 
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title(' Gray '), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(cv2.cvtColor(imgHarris, 
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title(' Harris '), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4)
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title(' Shi Tomasi '), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,5)
plt.imshow(cv2.cvtColor(imgORB, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title(' ORB '), plt.xticks([]), plt.yticks([])

plt.show()

#ORB BruteForce and Flann
plt.imshow(imgORBBruteForce),plt.show()
plt.imshow(imgORBFlann),plt.show()

#RGB Channels
#rows and colums for the plot
nrows = 2
ncols = 4

#Original
plt.subplot(nrows, ncols, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

#Display Red Channel
plt.subplot(nrows, ncols, 2)
plt.imshow(cv2.cvtColor(red, cv2.COLOR_BGR2RGB))
plt.title('Red'), plt.xticks([]), plt.yticks([])

#Display Green Channel
plt.subplot(nrows, ncols, 3)
plt.imshow(cv2.cvtColor(green, cv2.COLOR_BGR2RGB))
plt.title('Green'), plt.xticks([]), plt.yticks([])

#Display Blue Channel
plt.subplot(nrows, ncols, 4)
plt.imshow(cv2.cvtColor(blue, cv2.COLOR_BGR2RGB))
plt.title('Blue'), plt.xticks([]), plt.yticks([])

#HSV
plt.subplot(nrows, ncols, 5)
plt.imshow(cv2.cvtColor(imgHSV, cv2.COLOR_BGR2RGB))
plt.title('HSV'), plt.xticks([]), plt.yticks([])

#Hue
plt.subplot(nrows, ncols, 6)
plt.imshow(cv2.cvtColor(h_img, cv2.COLOR_BGR2RGB))
plt.title('Hue'), plt.xticks([]), plt.yticks([])

#Saturation
plt.subplot(nrows, ncols, 7)
plt.imshow(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB))
plt.title('Saturation'), plt.xticks([]), plt.yticks([])

#Value
plt.subplot(nrows, ncols, 8)
plt.imshow(cv2.cvtColor(v_img, cv2.COLOR_BGR2RGB))
plt.title('Value'), plt.xticks([]), plt.yticks([])

plt.show()

#Contour Detection
#rows and colums for the plot
nrows = 1
ncols = 2
plt.subplot(nrows, ncols,1)
plt.imshow(cv2.cvtColor(imgCD, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title(' Original '), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2)
plt.imshow(cv2.cvtColor(copy_imgCD, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title(' Contour Detection '), plt.xticks([]), plt.yticks([])
plt.show()