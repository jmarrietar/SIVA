# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 19:20:51 2016

@author: josemiguelarrieta
"""
import os 
import cv2
import numpy as np
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'


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
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(255,255,255),-1)

# salt coordinates
coords = [np.random.randint(0,512,25000), np.random.randint(0,512,25000)]

(B, G, R) = cv2.split(image)

# where does the salt coordinates land on the mask
a = mask[coords]
# find points where mask is 0 or 255
b, = np.nonzero(a==255)
# copy from coords only where mask is 0
valid_coords = np.array(coords)[:,b]
# apply salt on valid coordinates
R[valid_coords.tolist()]= 255
G[valid_coords.tolist()]= 255
B[valid_coords.tolist()]= 255

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

cv2.imwrite("../draft/Img_df_class1.jpg", merged)


cv2.destroyAllWindows()

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
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(255,255,255),-1)

# salt coordinates
coords = [np.random.randint(0,512,200000), np.random.randint(0,512,200000)]

(B, G, R) = cv2.split(image)

# where does the salt coordinates land on the mask
a = mask[coords]
# find points where mask is 0 or 255
b, = np.nonzero(a==255)
# copy from coords only where mask is 0
valid_coords = np.array(coords)[:,b]
# apply salt on valid coordinates
R[valid_coords.tolist()]= 0
G[valid_coords.tolist()]= 0
B[valid_coords.tolist()]= 0

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

cv2.imwrite("../draft/Img_df_class2.jpg", merged)

cv2.destroyAllWindows()

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
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(255,255,255),-1)

# salt coordinates
coords = [np.random.randint(0,512,100000), np.random.randint(0,512,100000)]

(B, G, R) = cv2.split(image)

# where does the salt coordinates land on the mask
a = mask[coords]
# find points where mask is 0 or 255
b, = np.nonzero(a==255)
# copy from coords only where mask is 0
valid_coords = np.array(coords)[:,b]
# apply salt on valid coordinates
R[valid_coords.tolist()]= 0
G[valid_coords.tolist()]= 0
B[valid_coords.tolist()]= 0

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

cv2.imwrite("../draft/Img_df_class3.jpg", merged)


cv2.destroyAllWindows()


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
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(255,255,255),-1)

# salt coordinates
coords = [np.random.randint(0,512,100000), np.random.randint(0,512,100000)]

(B, G, R) = cv2.split(image)

# where does the salt coordinates land on the mask
a = mask[coords]
# find points where mask is 0 or 255
b, = np.nonzero(a==255)
# copy from coords only where mask is 0
valid_coords = np.array(coords)[:,b]
# apply salt on valid coordinates
R[valid_coords.tolist()]= 128
G[valid_coords.tolist()]= 128
B[valid_coords.tolist()]= 128

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

cv2.imwrite("../draft/Img_df_class4.jpg", merged)

cv2.destroyAllWindows()


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
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(255,255,255),-1)

# salt coordinates
coords = [np.random.randint(0,512,100000), np.random.randint(0,512,100000)]

(B, G, R) = cv2.split(image)

# where does the salt coordinates land on the mask
a = mask[coords]
# find points where mask is 0 or 255
b, = np.nonzero(a==255)
# copy from coords only where mask is 0
valid_coords = np.array(coords)[:,b]
# apply salt on valid coordinates
R[valid_coords.tolist()]= 0
G[valid_coords.tolist()]= 0
B[valid_coords.tolist()]= 0

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

cv2.imwrite("../draft/Img_df_class5.jpg", merged)


cv2.destroyAllWindows()


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
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'],gt['y_position_centre']),(255,255,255),-1)

# salt coordinates
coords = [np.random.randint(0,512,200000), np.random.randint(0,512,200000)]

(B, G, R) = cv2.split(image)

# where does the salt coordinates land on the mask
a = mask[coords]
# find points where mask is 0 or 255
b, = np.nonzero(a==255)
# copy from coords only where mask is 0
valid_coords = np.array(coords)[:,b]
# apply salt on valid coordinates
R[valid_coords.tolist()]= 255
G[valid_coords.tolist()]= 255
B[valid_coords.tolist()]= 255

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

cv2.imwrite("../draft/Img_df_class6.jpg", merged)


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

