#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Label and Extract data
"""

import cv2
import numpy as np
import mahotas.features
from skimage.feature import hog
from localbinarypatterns import LocalBinaryPatterns
from balu.ImageProcessing import Bim_segbalu
from Bfx_basicint import Bfx_basicint


def get_data_SISL(cropped,cropped_maskA=None,cropped_maskB=None):
    """ 
    Labeling & Feature Extracion     
    
    Input:
        cropped = Cropped Section of Image. 
        cropped_maskA = Cropped Mask For Defect.
        cropped_maskB = Cropped Mask For Defect 2.
    Output: 
        If defect is AB return generalization of defect and respective defects as one. 
        Otherwise if defect is 1 or 2(One Defect) return instances and labels accordingly. 
    
    """
    (winW, winH) = (32, 32)                 #Descomposición en imágenes de 32 a 32 con corrimiento de a 32?. 
    labels = np.empty((0,1), int)
    instances =  np.empty((0,144), float)    #144 numero de Features
    insta_labelsA = np.empty((0,1), int)
    insta_labelsB = np.empty((0,1), int)
            
    #Check If image contains Defect 2
    if cropped_maskB is None:
        cropped_maskB = np.zeros(cropped.shape[:2], dtype = "uint8")
       
    #Check If image contains Defect 1
    if cropped_maskA is None:
        cropped_maskA = np.zeros(cropped.shape[:2], dtype = "uint8")
            
    #If image contains Defect 2 Record labels of defect
    if cropped_maskB is not None:
        insta_labelsB = np.empty((0,1), int)
        for (x, y, window_maskB,window) in sliding_window(cropped_maskB,cropped, stepSize=32, windowSize=(winW, winH)):
            #if the window does not meet our desired window size, ignore it
            if window_maskB.shape[0] != winH or window_maskB.shape[1] != winW:
                continue
            #Label defectB
            if (Defect(winW,winH,window_maskB)==True):
                insta_labelsB = np.append(insta_labelsB,np.array([[1]]), axis = 0)
            else:
                insta_labelsB = np.append(insta_labelsB,np.array([[0]]), axis = 0)
                
    #Do labeling and feature extraction         
    for (x, y, window_mask,window) in sliding_window(cropped_maskA,cropped, stepSize=32, windowSize=(winW, winH)):
        #if the window does not meet our desired window size, ignore it
        if window_mask.shape[0] != winH or window_mask.shape[1] != winW:
            continue
        if (Defect(winW,winH,window_mask)==True):
            insta_labelsA = np.append(insta_labelsA,np.array([[1]]), axis = 0)
        else:
            insta_labelsA = np.append(insta_labelsA,np.array([[0]]), axis = 0)
        #Extract Features
        instance = ExtractFeatures(window) 
        #instance.resize(1,len(instance))
        instances = np.append(instances,instance,axis=0)
        
    #If cropped_mask1 & cropped_mask2 exist then defect is AB
    #Si alguno de los windows tiene defecto, entonces es defectuoso. 
    if 255 in cropped_maskA  or 255 in cropped_maskB:
        for i in range(0,len(insta_labelsB)):
            if (insta_labelsA[i][0]==1 or insta_labelsB[i][0]==1):
                labels = np.append(labels,np.array([[1]]), axis = 0)
            else:
                labels = np.append(labels,np.array([[0]]), axis = 0)
        return labels,instances
    else:
        return insta_labelsA, instances
        
def get_data_MISL(cropped,cropped_maskA = None,cropped_maskB = None):
    """        
    Labeling & Feature Extracion  
    
    Input:
        cropped = Cropped Section of Image. 
        cropped_maskA = Cropped Mask For Defect.
        cropped_maskB = Cropped Mask For Defect 2 (default:None).
    Output: 
        If defect is AB return generalization of defect and respective defects as one. 
        Otherwise if defect is A or B(One Defect) return instances and labels accordingly.
    
    """
    if cropped_maskA is not None and cropped_maskB is not None: #Two Defect
        instance_labels,bag = get_data_SISL(cropped,cropped_maskA,cropped_maskB)
        if 1 in instance_labels:
            label_bag = 1
        else:
            label_bag = 0
        return label_bag, bag, instance_labels
    elif cropped_maskA is not None:   #One Defect
        instance_labels,bag = get_data_SISL(cropped,cropped_maskA)
        if 1 in instance_labels:
            label_bag = 1
        else:
            label_bag = 0
        return label_bag, bag, instance_labels 
    elif cropped_maskA is None and cropped_maskB is None:  #No Defect
        instance_labels,bag = get_data_SISL(cropped)
        return 0, bag, instance_labels

def get_data_SIML(cropped,cropped_maskA,cropped_maskB):
    """
    With SIML you can extract all combination from instance in Image with AB {0,1}{0,0}{1,0}{1,1}
    
    Input:
    
    Output: 
    
    Labeling & Feature Extracion 
    
    
    """
    
    instances =  np.empty((0,144), float)    #144 Number of Features
    insta_labelsA = np.empty((0,1), int)
    insta_labelsB = np.empty((0,1), int)
    (winW, winH) = (32, 32)                 #Descomposición en imágenes de 32 a 32 con corrimiento de a 32?. 
    
    #If defect is None
    if cropped_maskA is None: 
        cropped_maskA = np.zeros(cropped.shape[:2], dtype = "uint8")
    #If is just one defect
    if cropped_maskB is None: 
        cropped_maskB = np.zeros(cropped.shape[:2], dtype = "uint8")

    for (x, y, window_maskB,window) in sliding_window(cropped_maskB,cropped, stepSize=32, windowSize=(winW, winH)):
        #if the window does not meet our desired window size, ignore it
        if window_maskB.shape[0] != winH or window_maskB.shape[1] != winW:
            continue
        #Label defectA
        if (Defect(winW,winH,window_maskB)==True):
            insta_labelsB = np.append(insta_labelsB,np.array([[1]]), axis = 0)
        else:
            insta_labelsB = np.append(insta_labelsB,np.array([[0]]), axis = 0)
            
    for (x, y, window_maskA,window) in sliding_window(cropped_maskA,cropped, stepSize=32, windowSize=(winW, winH)):
        #if the window does not meet our desired window size, ignore it
        if window_maskA.shape[0] != winH or window_maskA.shape[1] != winW:
            continue     
        #Label defectB
        if (Defect(winW,winH,window_maskA)==True):
            insta_labelsA = np.append(insta_labelsA,np.array([[1]]), axis = 0)
        else:
            insta_labelsA = np.append(insta_labelsA,np.array([[0]]), axis = 0)
        #Extract Features        
        instance = ExtractFeatures(window)
        #instance.resize(1,len(instance))
        instances = np.append(instances,instance,axis=0)
    insta_labels = np.concatenate((insta_labelsA,insta_labelsB),axis=1)        
    return insta_labels, instances

def get_data_MIML(cropped,cropped_maskA,cropped_maskB):
    """
    
    With MIML is necesary images of defects AB, A, B and None
    
    Input:
    
    Output:
    
    """
    instance_labels, instances = get_data_SIML(cropped,cropped_maskA,cropped_maskB)
    instance_labelsA = instance_labels[:,[0]]
    instance_labelsB = instance_labels[:,[1]]
    bag = instances
    if 1 in instance_labelsA:
        labelA = 1
    else:
        labelA = 0
    if 1 in instance_labelsB:
        labelB = 1
    else:
        labelB = 0
    label_bag = np.array([[labelA,labelB]])
    return label_bag, bag, instance_labels


        

def sliding_window(image, image2 ,stepSize, windowSize):
    """
    Sliding windows implementation.    
    """
    # slide a window across the image
    for y in xrange(0, image2.shape[0], stepSize):
        for x in xrange(0, image2.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]],image2[y:y + windowSize[1], x:x + windowSize[0]])
            
def Defect(X,Y,window_mask):
    """
    Input:
    
    Output: 
    
    """
    if (float(np.count_nonzero(window_mask))/(X*Y)>0.10):
        return True
    else:
        return False
    
def ExtractFeatures (image):
    """
    Extract Features. 
        -> Features Extracted: 
            * LBP
            * Bfx_Basicint
            * Haralick
            * free Threshold Adjacency Statistics
            * Zernike Moments
            * HOG

    """    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #LBP Features
    desc = LocalBinaryPatterns(12, 4)
    lbp_hist = desc.describe(gray_image)
    lbp_hist = lbp_hist.reshape(1,len(lbp_hist))

    #Bfx_Basicint
    R,_,_ = Bim_segbalu(gray_image)
    options = {'show': True, 'mask': 5}
    basicint, Xn = Bfx_basicint(gray_image,R,options)

    #Haralick features
    haralick = mahotas.features.haralick(gray_image).mean(0)
    haralick = haralick.reshape(1,len(haralick))

    #parameter free Threshold Adjacency Statistics
    pftas = mahotas.features.pftas(gray_image)
    pftas = pftas.reshape(1,len(pftas))

    #Zernike Moments
    zernike = mahotas.features.zernike_moments(gray_image, radius=2)
    zernike = zernike.reshape(1,len(zernike))

    #HOG [Fix Dimentionality]
    HOG = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=False)
    HOG = HOG.reshape(1,len(HOG))

    #Join Features
    features = np.concatenate((lbp_hist,basicint,haralick,pftas,zernike,HOG), axis=1)
    
    return features