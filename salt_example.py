# -*- coding: utf-8 -*-
"""
Script to add salt and delete salt with Blurr

@author: josemiguelarrieta
"""
import cv2
import numpy as np
import os

##############
# Load image #
##############
# Load Image Example
path = '/Users/josemiguelarrieta/Documents/SIVA/'
filename = 'beach2.jpg'   
image = cv2.imread(path+filename)
cv2.imshow('image',image)

# Load Image defects
#Images Path 
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class' 
num = 150
filename = str(num)+'.PNG'
cl_number = 5 #Defect class
pathF = path+str(cl_number)+'_def'
image = cv2.imread(pathF+'/'+filename)
cv2.imshow('image',image)
cv2.destroyAllWindows()
##############################################

(B, G, R) = cv2.split(image)
coords = [np.random.randint(0,300,25000), np.random.randint(0,300,25000)]
valid_coords = np.array(coords)
valid_coords.tolist()
R[valid_coords.tolist()]= 0
G[valid_coords.tolist()]= 0
B[valid_coords.tolist()]= 0
merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)
blured = cv2.medianBlur(merged, 9)
cv2.imshow("blured", blured)
cv2.destroyAllWindows()



