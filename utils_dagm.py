# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:55:35 2017

@author: josemiguelarrieta
"""
import cv2
import numpy as np
import math

def load_image_dagm(path,num_image,class_number):
    filename = str(num_image)+'.PNG'
    pathF = path+str(class_number)+'_def'
    image = cv2.imread(pathF+'/'+filename)    
    return image

def load_labels_dagm(path,class_number,num):
    i = num - 1
    pathF = path+str(class_number)+'_def'
    f = open(pathF+'/'+'labels.txt', 'r')
    lines = f.readlines()
    gt = ground_truth_dagm(lines,i)
    return gt

def write_labels_defectA(cl_number,num,gt):
    """
    Ahora mismo esta quemada rutas y parametros [Modificar]
    """
    #Write labels_defect_A dimentions [Rectangle]
    target = open('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(cl_number)+'_defAB/'+'labels_defect_A.txt', 'a')
    line = str(num)+'\t'+ str(gt['semi_major_ax'])+'\t'+ str(gt['semi_minor_ax'])+'\t'+ str(int(gt['rotation_angle']))+'\t'+ str(gt['x_position_center'])+'\t'+ str(gt['y_position_center'])
    target.write(line)
    target.write("\n")
    target.close()
    return True
    
def write_labels_defectB(cl_number,num,d1,d2,d3,d4):
    #Write labels_defect_B dimentions [Rectangle]
    target = open('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(cl_number)+'_defAB/'+'labels_defect_B.txt', 'a')
    line = str(num)+'\t'+str(d1)+'\t'+str(d2)+'\t'+str(d3)+'\t'+str(d4)
    target.write(line)
    target.write("\n")
    target.close()
    return True

def rectangle_expanded_roi(gt):
    c1 = gt['x_position_center']
    c2 = gt['y_position_center']
    major_axis = gt['semi_major_ax']
    minus_axis =gt['semi_minor_ax']
    angle = gt['rotation_angle']
    ratation_angle_rad = np.deg2rad(angle)
    x = np.sqrt(major_axis**2*pow(np.cos(ratation_angle_rad),2) + minus_axis**2*pow(np.sin(ratation_angle_rad),2))
    y = np.sqrt(major_axis**2*pow(np.sin(ratation_angle_rad),2) + minus_axis**2*pow(np.cos(ratation_angle_rad),2))
    x1 = c1 - int(x)*2
    y1 = c2 - int(y)*2
    x2 = c1 + int(x)*2
    y2 = c2 + int(y)*2
    return x1, y1, x2, y2, x, y
    
def write_labels_expROI(cl_number,num,x1,y1,x2,y2):
    target = open('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(cl_number)+'_defAB/'+'ROI.txt', 'a')
    line = str(num)+'\t'+str(x1)+'\t'+str(y1)+'\t'+str(x2)+'\t'+str(y2)
    target.write(line)
    target.write("\n")
    target.close()
    return True

def defect_B_rect_ROI(x1, x, y1, y2):
    #Second defect rectangle dimentions
    d1 = x1
    d2 = y1
    d3 = x1+int(x)
    if y2 < y1+int(x)*2:
        d4 = y2
    else:
        d4 = y1+int(x)*2
    return d1, d2,  d3, d4    
    defect_B_rect_ROI(x1, x, y1, y2)

# Ellipse inside Rectangle [No rotation]
def ellipse_inside_rect(x1,y1,x,d4):
    x11 = x1
    y11 = y1
    x22 = x1 + int(x)
    y22 = d4
    B = x22 - x11
    A = y22 - y11
    c11 = B/2 + x11
    c22 = A/2 + y11
    return c11, c22, A, B

def add_salt(image,cl_number):
    (Blue, Green, Red) = cv2.split(image)
    length = len(image)
    number_salt = 25000    
    coords = [np.random.randint(0,length,number_salt), np.random.randint(0,length,number_salt)]
    valid_coords = np.array(coords)
    valid_coords.tolist()
    if cl_number == 5:
        color = 255
    else:
        color = 0
    Red[valid_coords.tolist()]= color
    Green[valid_coords.tolist()]= color
    Blue[valid_coords.tolist()]= color
    image_salted = cv2.merge([Blue, Green, Red])
    return image_salted

def add_blur(image2blur,cl_number): 
    #Select blur parameter acording to class number. 
    if cl_number == 1 or cl_number == 5:
        blur_parameter = 9
    elif cl_number == 2 or cl_number == 3:
        blur_parameter = 11
    else:
        blur_parameter = 13
    blured = cv2.medianBlur(image2blur, blur_parameter)
    return blured

def add_defect_B(c11, c22, A, B, blured, image):
    #Select ROI from image
    (cX, cY) = (blured.shape[1] // 2, blured.shape[0] // 2)
    mask = np.zeros(blured.shape[:2], dtype = "uint8")
    cv2.ellipse(mask, (c11,c22), (B/2,A/2),0,0,360,255, -1) 
    masked = cv2.bitwise_and(blured, blured, mask = mask)
    #cv2.imshow("Mask", mask)
    #cv2.imshow("Mask Applied to Image", masked)
    blured = masked
    img2 =  blured
    #create a ROI
    rows,cols,channels = img2.shape
    roi = image[0:rows, 0:cols]
    # Now create a mask  and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of interest from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    image[0:rows, 0:cols ] = dst
    return image

#Function save_image_created


def ground_truth_dagm (lines,line_number):
    line = lines[line_number].split("\t") #Change i to Numbers 
    number = int(line[0])
    semi_major_ax = int(float(line[1]))
    semi_minor_ax = int(float(line[2]))
    rotation_angle = math.degrees(float(line[3]))
    x_position_center = int(float(line[4]))
    y_position_center = int(float(line[5]))
    return {'number':number, 'semi_major_ax':semi_major_ax, 
            'semi_minor_ax':semi_minor_ax, 'rotation_angle':rotation_angle, 
            'x_position_center':x_position_center, 'y_position_center':y_position_center}
