# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:09:56 2017

@author: josemiguelarrieta
"""
#Load libraries 
import os 
import cv2
import mahotas.features
os.chdir('Documents/SIVA')
from localbinarypatterns import LocalBinaryPatterns
from balu.ImageProcessing import Bim_segbalu
from Bfx_basicint import Bfx_basicint
from skimage.feature import hog
import matplotlib.pyplot as plt

#Load Image
filename = 'Cl_1_2_AB.png'
image = cv2.imread(filename)

#convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#LBP Features
desc = LocalBinaryPatterns(12, 4)
lbp_hist = desc.describe(gray_image)

#Bfx_Basicint
R,_,_ = Bim_segbalu(gray_image)
options = {'show': True, 'mask': 5}
basicint, Xn = Bfx_basicint(gray_image,R,options)

#Haralick features
haralick = mahotas.features.haralick(gray_image).mean(0)

#parameter free Threshold Adjacency Statistics
pftas = mahotas.features.pftas(gray_image)

#Zernike Moments
zernike = mahotas.features.zernike_moments(gray_image, radius=2)

#HOG
hog = hog(gray_image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=False)

                    
#SURF
# Create SURF object. Hessian Threshold to 400
surf = cv2.SURF(400)
surf.upright = True
surf.extended = False

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(gray_image,None)

print len(kp)

# Find size of descriptor
print surf.descriptorSize()

img2 = cv2.drawKeypoints(gray_image,kp,None,(255,0,0),4)

plt.imshow(img2),plt.show()










