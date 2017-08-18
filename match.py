# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:01:36 2017

@author: Administrator
"""
import numpy as np
import cv2 


def des_distance(deep_des1,deep_des2):
    error = deep_des1-deep_des2
    RMSE = np.sqrt(np.sum(np.square(error),axis=1))/deep_des1.shape[0]
    
    return RMSE
    
def deep_match(kp1_location,kp2_location,deep_des1,deep_des2,ratio):
    deep_kp1 = []
    deep_kp2 = []
    for i in range(deep_des1.shape[0]):
        des = np.tile(deep_des1[i],(deep_des2.shape[0],1))
        error = des - deep_des2
        RMSE = np.sqrt(np.sum(np.square(error),axis=1)/error.shape[1])
        small_index = np.argsort(RMSE, axis=0)
        if RMSE[small_index[0]]< RMSE[small_index[1]]*ratio:
            deep_kp1.append((kp1_location[i][0],kp1_location[i][1]))
            deep_kp2.append((kp2_location[small_index[0]][0],kp2_location[small_index[0]][1]))
            #deep_des2 = np.delete(deep_des2, small_index[0], 0)
    return deep_kp1,deep_kp2

#match sift keypoints
def match(kp1_location,kp2_location,deep_des1,deep_des2,ratio):
    deep_kp1 = []
    deep_kp2 = []
    des1 = np.matrix(deep_des1)
    des2 = np.matrix(deep_des2)
    for i in range(des1.shape[0]):
        des1_ = np.tile(des1[i],(des2.shape[0],1))
        error = des1_ - des2
        RMSE = np.sqrt(np.sum(np.square(error),axis=1)/float(error.shape[1]))
        small_index = np.argsort(RMSE, axis=0)
        if RMSE[small_index[0,0],0] < RMSE[small_index[1,0],0]*ratio: 
            deep_kp1.append((kp1_location[i][0],kp1_location[i][1]))
            deep_kp2.append((kp2_location[small_index[0,0]][0],kp2_location[small_index[0,0]][1]))
            #deep_des2 = np.delete(deep_des2, small_index[0], 0)
    return deep_kp1,deep_kp2