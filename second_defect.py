# -*- coding: utf-8 -*-
"""
Create Second Defect on DAGM Images

"""
#Load libraries
import os 
import cv2

from utils_dagm import load_image_dagm, load_labels_dagm,write_labels_defectA, rectangle_expanded_roi, write_labels_expROI, defect_B_rect_ROI, write_labels_defectB, ellipse_inside_rect,add_salt,add_blur , add_defect_B

path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'
cl_number = 2       #Class number
num = 25            #File image number

#Load Image and Labels [Ground Truth] 
image = load_image_dagm(path,num,cl_number)
gt = load_labels_dagm(path,cl_number,num)

#cv2.ellipse(image,(gt['x_position_center'],gt['y_position_center']),(gt['semi_major_ax'],gt['semi_minor_ax']),gt['rotation_angle'],0,360,(0,255,0),2)  #Draw Ellipse [Ground Truth]

write_labels_defectA(cl_number,num,gt)            #Write labels defectA [Ground Truth]
x1, y1, x2, y2, x,y = rectangle_expanded_roi(gt)  #Dimentions of expanded rectangle 
#cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2) #Draw Rectangle

write_labels_expROI(cl_number,num,x1,y1,x2,y2)    #Write dimentions expanded rectangle [ROI]

d1, d2, d3, d4 = defect_B_rect_ROI(x1, x, y1, y2) #Second defect rectangle dimentions
#cv2.rectangle(image,(d1,d2),(d3,d4),(0,255,0),2) #Draw Rectangle defect B

write_labels_defectB(cl_number,num,d1,d2,d3,d4)   #Write labels defect B [Ground Truth]

c11, c22, A, B = ellipse_inside_rect(x1,y1,x,d4)  #Dimentions of Ellipse insed rectangle for defect B. 
#cv2.ellipse(image, (c11,c22), (B/2,A/2),0,0,360,(0,255,0),1) #Draw an Ellipse Below. 

image_salted = add_salt(image,cl_number)          #Added salt to Original image
blured = add_blur(image_salted,cl_number)         #Blured image salted
image4 = add_defect_B(c11, c22, A, B, blured, image) #Image with Final Defect B 
cv2.imshow('image4',image4)

cv2.destroyAllWindows()

################################
# SAVE: Save new Image Created #
################################

#Code Below
defect = 'AB'
filename_with_defect = str(num) + '_'+defect+'.PNG'
cv2.imwrite("../Class"+str(cl_number)+"_defAB/"+filename_with_defect, image)
