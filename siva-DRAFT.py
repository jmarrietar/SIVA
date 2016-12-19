# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 21:31:42 2016

@author: josemiguelarrieta
"""

from __future__ import print_function
import argparse
import cv2
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,help = "Path to the image")
args = vars(ap.parse_args())

cv2.startWindowThread()

image = cv2.imread("DSCF7940.JPG")
cv2.imshow("Image", image)
cv2.waitKey(0)


###Crop
cropped = image[300:1200 , 2400:3350]
cv2.imshow("T-Rex Face", cropped)
cv2.waitKey(0)

cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

canvas = np.zeros((300, 300, 3), dtype = "uint8")


green = (0, 255, 0)
cv2.line(canvas, (0, 0), (300, 300), green)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

red=(0,0,255)
cv2.line(canvas, (300, 0), (0, 300), red, 3)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

######
#SIFT#
######

import cv2
import numpy as np

img = cv2.imread("DSCF7940.JPG")
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('sift_keypoints.jpg',img)


img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

kp,des = sift.compute(gray,kp)

###########
# Masking #
###########
import cv2
import numpy as np

image = cv2.imread("beach.JPG")

#cv2.rectangle(image,(384,0),(510,128),(0,255,0),3)
cv2.ellipse(image,(256,256),(170,50),40,0,360,(0,255,0),2)
cv2.imshow("Original", image)

cv2.destroyAllWindows()

 (B, G, R) = cv2.split(image)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)

mask = np.zeros(image.shape[:2], dtype = "uint8")
(cX, cY) = (image.shape[1] / 2, image.shape[0] / 2)
cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75 , cY + 75), 255,-1)
cv2.imshow("Mask", mask)

masked = cv2.bitwise_and(image, image, mask = mask)

masked = cv2.bitwise_and(image,image, mask = mask)

cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)

(B, G, R) = cv2.split(masked)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)

cv2.destroyAllWindows()