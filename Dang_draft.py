# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 12:36:52 2016

@author: josemiguelarrieta
"""
from __future__ import print_function
import cv2
import numpy as np
import os
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure

##########################################################
#NOTE0:    Hacerlo en Sublime Text esta parte del Header #
##########################################################


##########################################################
path = '/Users/josemiguelarrieta/Downloads/Class1_def/'
os.chdir(path)

#####################################################
#NOTE_1:     Aqui Va el FOR de las Imagenes         #
#####################################################

#Load labels
f = open('labels.txt', 'r')
lines = f.readlines()
bags = []

for i in range(30,35):
    gt = ground_truth_dagm(lines,i)       # -> Number of File. 
    print(i)
    # -> Upload Dagm image. 
    filename = str(i+1)+'.PNG'                    # -> Filename number .PNG
    image = cv2.imread(filename)
    cv2.rectangle(image,(gt['x_position_centre'] - gt['semi_major_ax']*2,gt['y_position_centre'] - gt['semi_minor_ax']*2),(gt['x_position_centre'] + gt['semi_major_ax']*2,gt['y_position_centre'] + gt['semi_minor_ax']*2),(0,255,0),2)
    cv2.ellipse(image,(gt['x_position_centre'],gt['y_position_centre']),(gt['semi_major_ax'],gt['semi_minor_ax']),0.37,0,360,(0,255,0),2)
    cv2.imshow("Image"+str(i), image)
    #Create Mask of Image Uploaded!.
    (cX, cY) = (image.shape[1] / 2, image.shape[0] / 2)
    mask = np.zeros(image.shape[:2], dtype = "uint8")
    cv2.ellipse(mask,(gt['x_position_centre'],gt['y_position_centre']),(gt['semi_major_ax'],gt['semi_minor_ax']),0.37,0,360,255,-1)
    cv2.imshow("Mask"+str(i), mask)
    #cv2.destroyAllWindows()
    
    x1 = gt['y_position_centre'] - gt['semi_minor_ax']*2
    x2 = gt['y_position_centre'] + gt['semi_minor_ax']*2
    y1 = gt['x_position_centre'] - gt['semi_major_ax']*2
    y2 = gt['x_position_centre'] + gt['semi_major_ax']*2
    
    length,width,_ = image.shape
    
    start_x, end_x, start_y, end_y = get_cordinates_crop(x1,x2,y1,y2,length,width)
    
    #Week labeling (Cropping)[ROI] #Convertir esto en Funcion
    roi = image[start_x:end_x,start_y:end_y]   #Cropped de la imagen (rename to ROI)
    cv2.imshow("roi"+str(i), roi)

    roi_mask = mask[start_x:end_x,start_y:end_y]   #Cropped de la imagen (rename to ROI)
    cv2.imshow("roi_mask"+str(i), roi_mask)

    #Descomposición en imágenes de 32 a 32 con corrimiento de a 8. 
    (winW, winH) = (32, 32)

    instances = []
    insta_labels = []
    bag_label = []

    for (x, y, window_mask,window) in sliding_window(cropped_mask,cropped, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window_mask.shape[0] != winH or window_mask.shape[1] != winW:
            continue
        if (defect(winW,winH,window_mask)==True):
            insta_labels.append(1)
        else:
            insta_labels.append(0)
        print(defect(winW,winH,window_mask))
        window_gray = color.rgb2gray(window)
        #HOG Features
        fd, _ = hog(window_gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True, feature_vector = True)
        #LBP Features
        desc = LocalBinaryPatterns(24, 8)
        hist = desc.describe(window_gray)
        instance = np.concatenate((fd, hist), axis=0)
        instance.resize(1,len(instance))
        #Add Instance to Bag 
        instances.append(instance)
        #Con las Instancias Crear la Bolsa!.
    bag = np.asarray([instance[0] for instance in instances])
    bags.append(bag)   
        
#De aqui para abajo faltan cositas. 
        #List to array 
        if 1 in insta_labels:
            bag_label.append(1)
        else:
            bag_label.append(0)


def sliding_window(image, image2 ,stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image2.shape[0], stepSize):
		for x in xrange(0, image2.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]],image2[y:y + windowSize[1], x:x + windowSize[0]])

def defect(X,Y,window_mask): 
    if (float(np.count_nonzero(window_mask))/(X*Y)>0.10):
        return True
    else:
        return False

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

def get_cordinates_crop(x1,x2,y1,y2,length,width):    
    if (x1 < 0):
        start_x = 0
    elif (x1 > length):
        start_x = length
    else:
        start_x = x1
    if (x2 < 0):
        end_x = 0
    elif (x2 > width):
        end_x = width
    else:
        end_x = x2    
    if (y1 < 0):
        start_y = 0
    elif (y1 > length):
        y1 = length
    else:
        start_y = y1
    if (y2 < 0):
        end_y = 0
    elif (y2 > width):   
       y2 = width
    else:
        end_y = y2        
    return start_x, end_x, start_y, end_y