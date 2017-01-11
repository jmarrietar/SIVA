# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 20:38:50 2017

@author: josemiguelarrieta
"""

# Load image
import os 
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'
import cv2
import numpy as np

# -> Upload Dagm image. 
num = 2
filename = str(num)+'.PNG'   
i = num - 1

########
#Class1#
########
cl_number = 1
pathF = path+str(cl_number)+'_def'
os.chdir(pathF)
#Load labels
f = open('labels.txt', 'r')
lines = f.readlines()
gt = ground_truth_dagm(lines,i)

#MultiLabel
image = cv2.imread(filename)

x1 = gt['x_position_centre'] - gt['semi_major_ax']*2
y1 = gt['y_position_centre'] - gt['semi_minor_ax']*2
x2 = gt['x_position_centre']+gt['semi_major_ax']*2
y2 = gt['y_position_centre']+gt['semi_minor_ax']*2

cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 1)

#cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre']+gt['semi_major_ax']*2,gt['y_position_centre']+gt['semi_minor_ax']*2),(0,255,0),2)
#cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'] - gt['semi_major_ax']*2+40,gt['y_position_centre'] - gt['semi_minor_ax']*2+40),(0,255,0),2)
cv2.ellipse(image,(gt['x_position_centre'],gt['y_position_centre']),(gt['semi_major_ax'],gt['semi_minor_ax']),0.37,0,360,(0,255,0),2)  


######
#ROIS#
######

#ROI rectangulo Izquierda. 
cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'] - gt['semi_major_ax']*2+gt['semi_major_ax'],gt['y_position_centre'] - gt['semi_minor_ax']*2+gt['semi_major_ax']*2),(0,255,0),2)

x11 = gt['x_position_centre'] - gt['semi_major_ax']*2
y11 = gt['y_position_centre'] - gt['semi_minor_ax']*2
x22 = gt['x_position_centre'] - gt['semi_major_ax']*2+gt['semi_major_ax']
y22 = gt['y_position_centre'] - gt['semi_minor_ax']*2+gt['semi_major_ax']*2

###############################
# Elipsoid Based on Rectangle #
###############################
#Rectangle Dimentions

B = x22 - x11
A = y22 - y11

c11 = B/2 + x11
c22 = A/2 + y11


#Draw an Ellypse Below. 
cv2.ellipse(image, (c11,c22), (B/2,A/2),0.37,0,360,(0,255,0),1) 

#Elipsoid dimentions Based on Rectangle
##################################################


#ROI rectangulo arriba.
cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'] - gt['semi_major_ax']*2+gt['semi_major_ax']+60,gt['y_position_centre'] - gt['semi_minor_ax']*2+gt['semi_major_ax']),(0,255,0),2)

#Ellypse Inside Rectangle. 


#a = gt['x_position_centre'] - gt['semi_major_ax']*2
#b = gt['y_position_centre'] - gt['semi_minor_ax']*2
#c = gt['x_position_centre'] - gt['semi_major_ax']*2+gt['semi_major_ax']
#d = gt['y_position_centre'] - gt['semi_minor_ax']*2+gt['semi_major_ax']

cv2.imshow("Image", image)

cv2.destroyAllWindows()


####################################
#De Aqui a Abajo. ¿Que Voy a Hacer?.

#0) Blurrear Imagen Completa
image2blur = cv2.imread(filename)
cv2.imshow("Imagen", image)
blured = cv2.medianBlur(image2blur, 9)
cv2.imshow("Image Blurred", blured)


#Coger el roi de la imagen. 
(cX, cY) = (blured.shape[1] // 2, blured.shape[0] // 2)
mask = np.zeros(blured.shape[:2], dtype = "uint8")

cv2.ellipse(mask, (c11,c22), (B/2,A/2),0.37,0,360,255, -1) 
#cv2.rectangle(mask,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'] - gt['semi_major_ax']*2+gt['semi_major_ax'],gt['y_position_centre'] - gt['semi_minor_ax']*2+gt['semi_major_ax']*2), 255, -1)
#cv2.circle(mask, (cX, cY), 100, 255, -1)

masked = cv2.bitwise_and(blured, blured, mask = mask)

cv2.imshow("Mask", mask)
cv2.imshow("Mask Applied to Image", masked)

blured = masked


#Aplicarle las cosas de mesi 

#pegarlo encima
#img2 =  blur_resized
img2 =  blured
cv2.imshow("Image2", img2)

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = image[0:rows, 0:cols]
#cv2.imshow("roi_ini", roi)

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
#cv2.imshow("img1_bg", img1_bg)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
#cv2.imshow("img2_fg", img2_fg)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
cv2.imshow("dst", dst)
image[0:rows, 0:cols ] = dst
#cv2.imshow('res',image)

#cv2.waitKey(0)
cv2.destroyAllWindows()




def ground_truth_dagm (lines,line_number):
    line = lines[line_number].split("\t") #Change i to Numbers 
    number = int(line[0])
    semi_major_ax = int(float(line[1]))
    semi_minor_ax = int(float(line[2]))
    rotation_angle = int(float(line[3]))
    x_position_centre = int(float(line[4]))
    y_position_centre = int(float(line[5]))
    return {'number':number, 'semi_major_ax':semi_major_ax, 
            'semi_minor_ax':semi_minor_ax, 'rotation_angle':rotation_angle, 
            'x_position_centre':x_position_centre, 'y_position_centre':y_position_centre}
