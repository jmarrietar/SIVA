# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 19:26:40 2016

@author: josemiguelarrieta
"""
import os 
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'
import cv2
import numpy as np

# -> Upload Dagm image. 
filename = '2.PNG'   
i = 1

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
cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(105,105,105),4)
cv2.imshow("Image1", image)

cv2.destroyAllWindows()


#SingleLabel
image = cv2.imread(filename)
cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'] - gt['semi_major_ax']*2+50,gt['y_position_centre'] - gt['semi_minor_ax']*2+50),(105,105,105),4)
cv2.imshow("Image1", image)

cv2.destroyAllWindows()

cv2.imwrite("../draft/img_df_class1.jpg", image)



""""
#MASK
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(250,250,250),3)
cv2.imshow("Mask", mask)
""""


########
#Class2#
########
cl_number = 2
pathF = path+str(cl_number)+'_def'
os.chdir(pathF)
#Load labels
f = open('labels.txt', 'r')
lines = f.readlines()
gt = ground_truth_dagm(lines,i)

#MultiLabel
image = cv2.imread(filename)
cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(0,0,0),4)
cv2.imshow("Image1", image)


cv2.imwrite("../draft/img_df_class2.jpg", image)

cv2.destroyAllWindows()

#SingleLabel

########
#Class3#
########
cl_number = 3
pathF = path+str(cl_number)+'_def'
os.chdir(pathF)
#Load labels
f = open('labels.txt', 'r')
lines = f.readlines()
gt = ground_truth_dagm(lines,i)

#MultiLabel
image = cv2.imread(filename)
cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(255,255,255),1)
cv2.imshow("Image1", image)

cv2.destroyAllWindows()

cv2.imwrite("../draft/img_df_class3.jpg", image)


########
#Class4#
########
cl_number = 4
pathF = path+str(cl_number)+'_def'
os.chdir(pathF)
#Load labels
f = open('labels.txt', 'r')
lines = f.readlines()
gt = ground_truth_dagm(lines,i)

#MultiLabel
image = cv2.imread(filename)
cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(128,128,128),3)
cv2.imshow("Image1", image)

cv2.destroyAllWindows()

cv2.imwrite("../draft/img_df_class4.jpg", image)



########
#Class5#
########
cl_number = 5
pathF = path+str(cl_number)+'_def'
os.chdir(pathF)
#Load labels
f = open('labels.txt', 'r')
lines = f.readlines()
gt = ground_truth_dagm(lines,i)

#MultiLabel
image = cv2.imread(filename)
cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(0,0,0),1)
cv2.imshow("Image1", image)

cv2.destroyAllWindows()

cv2.imwrite("../draft/img_df_class5.jpg", image)


########
#Class6#
########
cl_number = 6
pathF = path+str(cl_number)+'_def'
os.chdir(pathF)
#Load labels
f = open('labels.txt', 'r')
lines = f.readlines()
gt = ground_truth_dagm(lines,i)

#MultiLabel
image = cv2.imread(filename)
cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(0,0,0),4)
cv2.imshow("Image1", image)

cv2.destroyAllWindows()

cv2.imwrite("../draft/img_df_class6.jpg", image)



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

