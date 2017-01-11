# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:48:57 2017

@author: josemiguelarrieta

"""
import os 
import cv2
import numpy as np

path = '/Users/josemiguelarrieta/Documents/SIVA/'
os.chdir(path)

# Load two images
img1 = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv-logo.png')

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
cv2.imshow("roi_ini", roi)
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow("img1_bg", img1_bg)
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
cv2.imshow("img2_fg", img2_fg)
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
cv2.imshow("dst", dst)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()



##########################
##########################

#######
#STEPS#
#######


#Load Imagen Original. 
# -> Upload Dagm image. 
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'
# -> Upload Dagm image. 
filename = '2.PNG'   
i = 1
cl_number = 1
#cl_number = 2

#NO BLURRES LA CROP, BLURREA LA IMAGEN DE COPIA

#BLURREA


#LUEGO CROP

pathF = path+str(cl_number)+'_def'
os.chdir(pathF)

image = cv2.imread(filename)

image2blur = cv2.imread(filename)

cv2.imshow("Imagen", image)

#Blur Imagen.   # Hacerle un ShowImage. 
blured = cv2.medianBlur(image2blur, 9)
cv2.imshow("Image Blurred", blured)

###############################################
# IMPORTANTE AQUI SELECCIONAR UN PARCHE        #
# Cambiando Aleatoriamente sus dimenciones    #
##############################################

# Select circulo.
(cX, cY) = (blured.shape[1] // 2, blured.shape[0] // 2)
mask = np.zeros(blured.shape[:2], dtype = "uint8")

cv2.circle(mask, (cX, cY), 100, 255, -1)

masked = cv2.bitwise_and(blured, blured, mask = mask)

cv2.imshow("Mask", mask)
cv2.imshow("Mask Applied to Image", masked)

blured = masked

"""
#Esto del Sized es Talves No Necesario!.
r = 150.0 / blured.shape[1]
dim = (150, int(blured.shape[0] * r))

blur_resized = cv2.resize(blured, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resized (Width)", blur_resized)
"""

#ombe tener la imagen con la mascara aplicarsela. 

#pegarlo encima
#img2 =  blur_resized
img2 =  blured
cv2.imshow("Image2", img2)

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = image[0:rows, 0:cols ]
cv2.imshow("roi_ini", roi)
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow("img1_bg", img1_bg)
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
cv2.imshow("img2_fg", img2_fg)
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
cv2.imshow("dst", dst)
image[0:rows, 0:cols ] = dst
cv2.imshow('res',image)
#cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Img_df_class6.jpg", image)


#Final. Ponerlo en el Lugar Exactamente que Yo Quiero. 
