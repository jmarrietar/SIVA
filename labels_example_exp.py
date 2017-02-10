# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:16:23 2017

@author: josemiguelarrieta
"""
#Load libraries
import os 
os.chdir('Documents/SIVA')
import cv2

from utils_dagm import get_labels_defectA, get_labels_defectB, get_roi_rect, load_image_dagm

###################################
#Check Labeling for Experiment #1 #
###################################

class_number = 2
number_experimet = 1
num=1
path= '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'


##########
#Defect A#
##########

#Load image
image1 = load_image_dagm(path,num,class_number,defect = 'A',exp = True)
#get labels
dfA_A = get_labels_defectA(path,class_number,num,exp=True,defect='A',degrees=True)
cv2.ellipse(image1,(dfA_A['x_position_center'],dfA_A['y_position_center']),(dfA_A['semi_major_ax'],dfA_A['semi_minor_ax']),dfA_A['rotation_angle'],0,360,(0,255,0),2)  #Draw Ellipse [Ground Truth]
roi_A = get_roi_rect(path,class_number,num,exp=True,defect='A')
cv2.rectangle(image1,(roi_A['x1'],roi_A['y1']),(roi_A['x2'],roi_A['y2']),(0,255,0),2)
cv2.imshow('image1'+str(num),image1)

##########
#Defect B#
##########
#Load image
image2 = load_image_dagm(path,num,class_number,defect = 'B',exp = True)
#get labels
dfB_B = get_labels_defectB(path,class_number,num,exp=True,defect='B')
roi_B = get_roi_rect(path,class_number,num,exp=True,defect='B')
cv2.rectangle(image2,(roi_B['x1'],roi_B['y1']),(roi_B['x2'],roi_B['y2']),(0,255,0),2)
cv2.rectangle(image2,(dfB_B['x1'],dfB_B['y1']),(dfB_B['x2'],dfB_B['y2']),(0,255,0),2)
cv2.imshow('image2'+str(num),image2)

##########
#Defect AB#
##########
#Load image
image3 = load_image_dagm(path,num,class_number,defect = 'AB',exp = True)
#get labels
dfAB_A = get_labels_defectA(path,class_number,num,exp=True,defect='AB',degrees=True)
dfAB_B = get_labels_defectB(path,class_number,num,exp=True,defect='AB')
roi_AB = get_roi_rect(path,class_number,num,exp=True,defect='AB')
cv2.rectangle(image3,(roi_AB['x1'],roi_AB['y1']),(roi_AB['x2'],roi_AB['y2']),(0,255,0),2)
cv2.rectangle(image3,(dfAB_B['x1'],dfAB_B['y1']),(dfAB_B['x2'],dfAB_B['y2']),(0,255,0),2)
cv2.ellipse(image3,(dfAB_A['x_position_center'],dfAB_A['y_position_center']),(dfAB_A['semi_major_ax'],dfAB_A['semi_minor_ax']),dfAB_A['rotation_angle'],0,360,(0,255,0),2)  #Draw Ellipse [Ground Truth]
cv2.imshow('image3'+str(num),image3)


cv2.destroyAllWindows() 
