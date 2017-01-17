# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:11:32 2017

@author: josemiguelarrieta
"""
import os 
import cv2
import numpy as np
os.chdir('Documents/SIVA')
from localbinarypatterns import LocalBinaryPatterns


path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'

# -> Upload Dagm image. 
filename = '2.PNG'   
i = 1

########
#Class1#
########
cl_number = 1
pathF = path+str(cl_number)+'_def'
os.chdir(pathF)

#MultiLabel
image = cv2.imread(filename)

# load the image, convert it to grayscale, and describe it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#LBP Features
desc = LocalBinaryPatterns(12, 4)
hist = desc.describe(gray)