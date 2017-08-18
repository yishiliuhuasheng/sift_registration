# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:26:14 2017

@author: Administrator
"""
import cv2
import numpy as np

import match
import display
import ransac
import image_fusion
import MI


error_threshold = 1
sift_ratio = 0.7                    #knn1/lnn2 
block = 8
threshold = 300
#read image
img1 = cv2.imread("testdata/w1.jpg") # image to be registered
img2 = cv2.imread("testdata/w2.jpg") # reference image

#rgb2gray
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

kp1_location = []
kp2_location = []
kp1_angle = []
kp2_angle = []

for i in range(len(kp1)):
    kp1_location.append(kp1[i].pt)
    kp1_angle.append(kp1[i].angle)

for i in range(len(kp2)):
    kp2_location.append(kp2[i].pt)
    kp2_angle.append(kp2[i].angle)    

good_kp1,good_kp2 = match.match(kp1_location,kp2_location,des1,des2,sift_ratio)
img_good = display.display(img1,img2,good_kp1,good_kp2)
better_kp1,better_kp2 = ransac.ransac(good_kp1,good_kp2,error_threshold)

solution,rmse = ransac.least_square(better_kp1,better_kp2)
img_better = display.display(img1,img2,better_kp1,better_kp2)
sift_fusion = image_fusion.image_fusion(img1,img2,solution)


common1,common2 = image_fusion.common_region(gray1,gray2,solution)
mi = MI.MI(common1,common2)
print mi

cv2.imshow("sift1 image good match", img_good) #show mathces
cv2.imshow("sift1 image better match", img_better) #show mathces
cv2.imshow("sift1 image fusion", sift_fusion) #show fusion   
cv2.waitKey(0)                           
                
            
                                             





