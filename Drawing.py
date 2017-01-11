# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:07:30 2017

@author: josemiguelarrieta
"""

import numpy as np
import cv2

##########################################################################
#NOTE IMPORTANT: * DO NOT DELETE                                         #
#                * Convert This to Ipython NoteBook/Blog/Easy Tutorial   #
##########################################################################


#Rectangle Dimentions
x1 = 30
y1 = 30

x2 = 45
y2 = 75

B = x2 - x1
A = y2 - y1

c1 = B/2 + x1
c2 = A/2 + y1

canvas = np.zeros((300, 300, 3), dtype = "uint8")

cv2.rectangle(canvas, (x1, y1), (x2, y2), red, 1)
#Draw an Ellypse Below. 
cv2.ellipse(canvas, (c1,c2), (B/2,A/2),0.37,0,360,(0,255,0),1) 

cv2.imshow("Canvas", canvas)
cv2.destroyAllWindows()

#Elipsoid dimentions Based on Rectangle


