# -*- coding: utf-8 -*-
"""
@author: josemiguelarrieta
"""

#Image Manipulation
from __future__ import print_function
import os
os.chdir('Documents/SIVA')
import cv2
import numpy as np
from utils_dagm import WeakLabeling, get_data_SISL, get_data_SIML, get_data_MISL, get_data_MIML
from utils_dagm import sliding_window,defect
from skimage.feature import hog
from skimage import color
from localbinarypatterns import LocalBinaryPatterns

#Image Information [1 Image Only]
class_number = 1
number_experimet = 1
num = 1
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'

#WeakLabeling

#WeakLabeling defect A
cropped,cropped_mask = WeakLabeling(path,num,class_number,defect = 'A',exp = True)

cv2.imshow("cropped", cropped)
cv2.imshow("cropped_mask", cropped_mask)

#WeakLabeling defect AB [MULTILABEL]
cropped,cropped_maskA,cropped_maskB = WeakLabeling(path,num,class_number,defect = 'AB',exp = True)

cv2.imshow("cropped", cropped)
cv2.imshow("cropped_maskA", cropped_maskA)
cv2.imshow("cropped_maskB", cropped_maskB)
cv2.destroyAllWindows()

#SISL [1 imagen]
labels_A,instances_A = get_data_SISL_draft(cropped,cropped_maskA)
labels_B,instances_B = get_data_SISL_draft(cropped,cropped_maskA)
labels_AB,instances_AB = get_data_SISL_draft(cropped,cropped_maskA,cropped_maskB)
                
#MISL[1 imagen]
label_bag, bag = get_data_MISL(cropped,cropped_mask)
         
#SIML[1 imagen]
insta_labels, instances = get_data_SIML(cropped,cropped_maskA,cropped_maskB)
insta_labels, instances = get_data_SIML(cropped,cropped_maskA)
            
#MIML[1 imagen]
labels_bag, bag = get_data_MIML(cropped,cropped_maskA,cropped_maskB)

    
def get_data_SISL_draft(cropped,cropped_maskA,cropped_maskB=None):
    """
    Funciona para defectA y defectB, pero No esta Generalizado para defectAB. 
    Input:
    
    Output: 

    Labeling & Feature Extracion
    
    """
    (winW, winH) = (32, 32)                 #Descomposición en imágenes de 32 a 32 con corrimiento de a 32?. 
    labels = np.empty((0,1), int)
    instances =  np.empty((0,58), float)    #Cambiar 58 por numero de Features
    insta_labelsA = np.empty((0,1), int)
    
    if cropped_maskB is not None:
        insta_labelsB = np.empty((0,1), int)
        for (x, y, window_maskB,window) in sliding_window(cropped_maskB,cropped, stepSize=32, windowSize=(winW, winH)):
            #if the window does not meet our desired window size, ignore it
            if window_maskB.shape[0] != winH or window_maskB.shape[1] != winW:
                continue
            #Label defectB
            if (defect(winW,winH,window_maskB)==True):
                insta_labelsB = np.append(insta_labelsB,np.array([[1]]), axis = 0)
            else:
                insta_labelsB = np.append(insta_labelsB,np.array([[0]]), axis = 0)
    for (x, y, window_mask,window) in sliding_window(cropped_maskA,cropped, stepSize=32, windowSize=(winW, winH)):
        #if the window does not meet our desired window size, ignore it
        if window_mask.shape[0] != winH or window_mask.shape[1] != winW:
            continue
        if (defect(winW,winH,window_mask)==True):
            insta_labelsA = np.append(insta_labelsA,np.array([[1]]), axis = 0)
        else:
            insta_labelsA = np.append(insta_labelsA,np.array([[0]]), axis = 0)
            
        window_gray = color.rgb2gray(window)        
        #HOG [Fix Dimentionality]
        fd, _ = hog(window_gray, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True, feature_vector = True)
        #LBP Features
        desc = LocalBinaryPatterns(24, 8)
        hist = desc.describe(window_gray)        
        instance = np.concatenate((fd, hist), axis=0)
        instance.resize(1,len(instance))
        instances = np.append(instances,instance,axis=0)
        
    if cropped_maskB is not None:
        for i in range(0,len(insta_labelsB)):
            if (insta_labelsA[i][0]==1 or insta_labelsB[i][0]==1):
                labels = np.append(labels,np.array([[1]]), axis = 0)
            else:
                labels = np.append(labels,np.array([[0]]), axis = 0)
        return labels,instances
    else:
        return insta_labelsA, instances