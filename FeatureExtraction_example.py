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
import numpy as np
from mahotas.features import surf
from pylab import imshow,show


#Load Image
filename = 'Cl_1_2_AB.png'
image = cv2.imread(filename)

#convert image to grayscale
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
hog = hog(gray_image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=False)
hog = hog.reshape(1,len(hog))

#SURF
colors = np.array([(0,255,0)])
l, w = gray_image.shape
a = np.array([l/2,w/2,14,54,1])
a = a.reshape(1,len(a))
sp2 = surf.descriptors(gray_image, a, is_integral=False, descriptor_only=False)
sp3 = surf.descriptors(gray_image, a, is_integral=False, descriptor_only=True)
f2 = surf.show_surf(gray_image, sp2,colors = colors)
imshow(f2)
show()

#Join Features
features = np.concatenate((lbp_hist,basicint,haralick,pftas,zernike,hog,sp3), axis=1)