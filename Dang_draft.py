# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 12:36:52 2016

@author: josemiguelarrieta
"""
from __future__ import print_function
import argparse
import cv2
import numpy as np
import os

########################################################
#Objetivo: 

###
#1#
###
# - Subir Imagen de dang. 
path = '/Users/josemiguelarrieta/Downloads/Class1_def/'
os.chdir(path)
filename = '1.PNG'
image = cv2.imread(filename)
#cv2.rectangle(image,(69-60,87-37),(69-60+60+60,87-37+37+37),(0,255,0),2)
cv2.rectangle(image,(69-60-60,87-37-37),(69-60+60+60+60,87-37+37+37+37),(0,255,0),2)
cv2.ellipse(image,(69,87),(60,37),0.65,0,360,(0,255,0),2)
cv2.imshow("Image", image)

###
#2#
###
# - Subir Imagen de dang. 
path = '/Users/josemiguelarrieta/Downloads/Class1_def/'
os.chdir(path)
filename = '2.PNG'
image = cv2.imread(filename)
#cv2.rectangle(image,(315-41,172-35),(315-41+41+41,172-35+35+35),(0,255,0),2)
cv2.rectangle(image,(315-41-41,172-35-35),(315-41+41+41+41,172-35+35+35+35),(0,255,0),2)
cv2.ellipse(image,(315,172),(41,35),0.37,0,360,(0,255,0),2)
cv2.imshow("Image", image)

#Descomposición en imágenes de 16 a 16 con corrimiento de a 8. 

cv2.destroyAllWindows()

# Hacer el week labeling (Cropping)

cropped = image[102:233 , 242:397]
cv2.imshow("T-Rex Face", cropped)

cv2.waitKey(0)

window = cropped[0:16 , 0:16]

######
#SIFT#
######
gray= cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('sift_keypoints.jpg',img)

img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

kp,des = sift.compute(gray,kp)


#Extraer caracteristicas,  con Ventana? Como ?. 





"""
label = cv2.imread('Label/0595_label.PNG')
cv2.imshow("Label", label)
(B, _, _) = cv2.split(label)

masked = cv2.bitwise_and(image, image, mask = B)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)

cv2.startWindowThread()
cv2.destroyAllWindows()

"""

