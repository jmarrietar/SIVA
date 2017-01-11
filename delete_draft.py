# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 19:02:51 2017

@author: josemiguelarrieta
"""

import os 
import cv2
import numpy as np

path = '/Users/josemiguelarrieta/Documents/SIVA/'
os.chdir(path)

filename='beach2.jpg'
image = cv2.imread(path+filename)
cv2.imshow("Image", image)


blurred = np.hstack([
    cv2.medianBlur(image, 3),
    cv2.medianBlur(image, 5),
    cv2.medianBlur(image, 7)])
cv2.imshow("Median", blurred)

roi = image[0:200,0:200]
roi2  = cv2.medianBlur(roi, 9)

cv2.imshow("roi", roi)
cv2.imshow("roi2", roi2)

#Insert image into another
import cv2
s_img = roi2
l_img = image
x_offset=y_offset=50
l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

cv2.imshow("imag_patch", l_img)

cv2.destroyAllWindows()



################################################################
################################################################
################################################################

###########################
#Crea Patches con Blurr!. #
###########################
import os 
import cv2
import numpy as np

#Upload Texture. 
# -> Upload Dagm image. 
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'

# -> Upload Dagm image. 
filename = '2.PNG'   
i = 1
cl_number = 1
pathF = path+str(cl_number)+'_def'
os.chdir(pathF)

image = cv2.imread(filename)

cv2.imshow("Imagen", image)

roi_d = image[50:200,50:200]

#OK. 
#AQUI TOMASTE UN PARCHE. OK?. BIEN. 
#Entonces, lo que debes hacer ahora es esto!. 
#DE LA IMAGEN BLURRED ESA
#VAS A CREAR UN  UNA MASCARA



cv2.imshow("roi_d", roi_d)

roid2 = cv2.medianBlur(roi_d, 9)

image[y_offset:y_offset+roid2.shape[0], x_offset:x_offset+roid2.shape[1]] = roid2

cv2.imshow("imag_patch", image)

cv2.imshow("roid2", roid2)



#NOTE IMPORTANTE: Aun no he hecho lo de 

################################################
#FALTA: Me Falta CREAR UNA MASCARA IRREGULAR. mmmm


######################################
#Aqui voy a Hacer lo de la Mascara!. #
######################################

(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)

mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.circle(mask, (cX, cY), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask = mask)

(B, G, R) = cv2.split(masked)

cv2.imshow("Mask", mask)
cv2.imshow("Mask Applied to Image", masked)

cv2.waitKey(0





