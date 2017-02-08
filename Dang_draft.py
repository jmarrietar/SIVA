# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 12:36:52 2016

@author: josemiguelarrieta
"""

##########################################################
#NOTE0:    Hacerlo en Sublime Text esta parte del Header #
##########################################################

                                #------------------------#
                                #-- Image Manipulation --#
                                #------------------------#
from __future__ import print_function
import os
os.chdir('Documents/SIVA')
import cv2
import numpy as np
from utils_dagm import load_image_dagm, get_roi_rect, get_labels_defectA, get_labels_defectB
from skimage.feature import hog
from skimage.feature import hog
from skimage import color
from localbinarypatterns import LocalBinaryPatterns

#Image Information [1 Image Only]
class_number = 1
number_experimet = 1
num = 1
path= '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'

cropped,cropped_mask = WeakLabeling(path,num,class_number,defect = 'A',exp = True)
#IMPORTANT: 
#Nota WeekLabeling Le falta Opcion para AB!

cv2.imshow("cropped", cropped)
cv2.imshow("cropped_mask", cropped_mask)
cv2.destroyAllWindows()

##############################################################               

#SISL [1 imagen]
labels,instances = get_data_SISL(cropped,cropped_mask)

    

###############################################################               
#MISL[1 imagen]
label_bag, bag = get_data_MISL(cropped,cropped_mask)

        ############################
        #MULTIPLE INSTANCE LEARNING# 
        ############################
#Crear un array que guarde los IDS de las instancias.
        
#Aqui crear un array con el id de la Bolsa. 
###############################################################
                ######                
                #SIML#
                ######

#DEMASIADO IMPORTANTE: Aqui en ML, HACER 2 MASK PORQUE TIENE 2 DEFECTOS. 
 #SIML [Hacer en otro archivo/ Necesitas las 2 labels (2 gt A Y B defect)]


#nota

################################################################

###############################################################
                ######                
                #MIML#
                ######

#DEMASIADO IMPORTANTE: Combinar MI y ML!!!!!!. 
 #MIML [Hacer en otro archivo/ Necesitas las 2 labels (2 gt A Y B defect)]

################################################################

def get_data_SISL(cropped,cropped_mask):
    """
    
    Labeling & Feature Extracion
    
    """
    instances =  np.empty((0,58), float)    #Cambiar 58 por numero de Features
    insta_labels = np.empty((0,1), int)
    (winW, winH) = (32, 32)                 #Descomposición en imágenes de 32 a 32 con corrimiento de a 32?. 
    for (x, y, window_mask,window) in sliding_window(cropped_mask,cropped, stepSize=32, windowSize=(winW, winH)):
        #if the window does not meet our desired window size, ignore it
        if window_mask.shape[0] != winH or window_mask.shape[1] != winW:
            continue
        if (defect(winW,winH,window_mask)==True):
            insta_labels = np.append(insta_labels,np.array([[1]]), axis = 0)
        else:
            insta_labels = np.append(insta_labels,np.array([[0]]), axis = 0)
        window_gray = color.rgb2gray(window)
        
        #HOG [Fix Dimentionality]
        fd, _ = hog(window_gray, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True, feature_vector = True)
        #LBP Features
        desc = LocalBinaryPatterns(24, 8)
        hist = desc.describe(window_gray)
        
        instance = np.concatenate((fd, hist), axis=0)
        instance.resize(1,len(instance))
        instances = np.append(instances,instance,axis=0)
    return insta_labels, instances
    
def get_data_MISL(cropped,cropped_mask):
    labels,bag = get_data_SISL(cropped,cropped_mask)
    if 1 in labels:
        label_bag = 1
    else:
        label_bag = 0
    return label_bag, bag

def defect(X,Y,window_mask): 
    if (float(np.count_nonzero(window_mask))/(X*Y)>0.10):
        return True
    else:
        return False

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

def sliding_window(image, image2 ,stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image2.shape[0], stepSize):
		for x in xrange(0, image2.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]],image2[y:y + windowSize[1], x:x + windowSize[0]])

def WeakLabeling(path,num,class_number,defect = 'X',exp = True):
    """
    Importante: Falta Implementar para Defect AB!
    """
    image = load_image_dagm(path,num,class_number,defect = defect,exp = True)
    length, width,_ = image.shape
    #Create Mask of Image Uploaded!.
    mask = np.zeros(image.shape[:2], dtype = "uint8") 
    #get roi
    roi_labels = get_roi_rect(path,class_number,num,exp=True,defect=defect)
    if defect=='A': 
        #get labels defectA
        dfA_A = get_labels_defectA(path,class_number,num,exp=True,defect='A',degrees=True)    
        #Draw Defect on Mask
        cv2.ellipse(mask,(dfA_A['x_position_center'],dfA_A['y_position_center']),(dfA_A['semi_major_ax'],dfA_A['semi_minor_ax']),dfA_A['rotation_angle'],0,360,255,-1)  #Draw Ellipse [Ground Truth]
    elif defect=='B':
        #get labels defectBs
        dfB_B = get_labels_defectB(path,class_number,num,exp=True,defect='B')
        #Draw Defect on Mask
        cv2.rectangle(mask,(dfB_B['x1'],dfB_B['y1']),(dfB_B['x2'],dfB_B['y2']),255,-1)
    else:
        return 0,0
    #Week labeling [Cropping-ROI]
    start_x, end_x, start_y, end_y = get_cordinates_crop(roi_labels['x1'],roi_labels['x2'],roi_labels['y1'],roi_labels['y2'],length,width)
    roi = image[start_y:end_y,start_x:end_x]      
    roi_mask = mask[start_y:end_y,start_x:end_x]
    return roi,roi_mask

"""
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
"""