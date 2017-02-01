# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:55:35 2017

@author: josemiguelarrieta
"""
import cv2
import numpy as np
import math
import pickle
from shutil import copyfile

def add_salt(image,cl_number):
    """
    Input:
    
    Output:
    
    """
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
    """
    Input: 
    
    Output: 
    
    """
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
    """
    Input: 
    
    Output: 
    
    """
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

def defect_B_rect_ROI(x1, x, y1, y2):
    """
    Input: 
    
    Output: 
    
    """
    #Second defect rectangle dimentions
    d1 = x1
    d2 = y1
    d3 = x1+int(x)
    if y2 < y1+int(x)*2:
        d4 = y2
    else:
        d4 = y1+int(x)*2
    return d1, d2,  d3, d4

# Ellipse inside Rectangle [No rotation]
def ellipse_inside_rect(x1,y1,x,d4):
    """
    Input: 
    
    Output: 
    
    """
    x11 = x1
    y11 = y1
    x22 = x1 + int(x)
    y22 = d4
    B = x22 - x11
    A = y22 - y11
    c11 = B/2 + x11
    c22 = A/2 + y11
    return c11, c22, A, B
    
def get_labels_defectA(path,class_number,num,exp='',defect='',degrees=False):
    i = num - 1
    if exp == True:
        if defect == 'A' or defect =='AB':
            pathF = path+str(class_number)+'/'+'Cl_'+str(class_number)+'_'+defect
            f = open(pathF+'/'+'labels_defect_A.txt', 'r')
    else: 
        pathF = path+str(class_number)+'_def'
        f = open(pathF+'/'+'labels.txt', 'r')
    lines = f.readlines()
    gt = ground_truth_defectA(lines,i,degrees=degrees)
    return gt
              
def get_labels_defectB(path,class_number,num,exp='',defect=''):
    """
    Input: 
    
    Output: 
    
    """
    i = num - 1
    if exp == True:
        if defect == 'B' or defect =='AB':
            pathF = path+str(class_number)+'/'+'Cl_'+str(class_number)+'_'+defect
            f = open(pathF+'/'+'labels_defect_B.txt', 'r')
    else: 
        pathF = path+str(class_number)+'_defAB'
        f = open(pathF+'/'+'labels_defect_B.txt', 'r')
    lines = f.readlines()
    gt = ground_truth_defectB(lines,i)
    return gt
    
def get_roi_rect(path,class_number,num,exp='',defect=''):
    line_number = num -1 
    if exp == True:
        pathF = path+str(class_number)+'/'+'Cl_'+str(class_number)+'_'+defect
        f = open(pathF+'/'+'ROI.txt', 'r')  
    else:
        pathF = path+str(class_number)+'_defAB'   
        f = open(pathF+'/'+'ROI.txt', 'r')
    lines = f.readlines()
    line = lines[line_number].split("\t") 
    number = int(line[0])
    x1 = int(float(line[1]))
    y1 = int(float(line[2]))
    x2 = int(float(line[3]))
    y2 = int(float(line[4]))
    return {'number':number, 'x1':x1, 
            'y1':y1, 'x2':x2, 
            'y2':y2}
    
def ground_truth_defectB(lines,line_number):
    """
    Input: 
    
    Output: 
    
    """
    line = lines[line_number].split("\t") #Change i to Numbers 
    number = int(line[0])
    x1 = int(float(line[1]))
    y1 = int(float(line[2]))
    x2 = int(float(line[3]))
    y2 = int(float(line[4]))
    return {'number':number, 'x1':x1, 
            'y1':y1, 'x2':x2, 
            'y2':y2}
    
def ground_truth_defectA (lines,line_number,degrees=False):
    """
    Input: 
    
    Output: 
    
    """
    line = lines[line_number].split("\t") #Change i to Numbers 
    number = int(line[0])
    semi_major_ax = int(float(line[1]))
    semi_minor_ax = int(float(line[2]))
    if degrees==True:
        rotation_angle = int(line[3])
    else:
        rotation_angle = math.degrees(float(line[3]))
    x_position_center = int(float(line[4]))
    y_position_center = int(float(line[5]))
    return {'number':number, 'semi_major_ax':semi_major_ax, 
            'semi_minor_ax':semi_minor_ax, 'rotation_angle':rotation_angle, 
            'x_position_center':x_position_center, 'y_position_center':y_position_center}
    
def load_image_dagm(path,num_image,class_number,defect = '_def',exp = ''):
    """
    Input: 
    Output: 
    """
    if exp == True:
        pathF = path+str(class_number)+'/Cl_'+str(class_number)+'_'+defect
        filename = 'Cl_'+str(class_number)+'_'+str(num_image)+'_'+defect+'.png'
        image = cv2.imread(pathF+'/'+filename)
    else:
        filename = str(num_image)+'.png'
        pathF = path+str(class_number)+defect
        image = cv2.imread(pathF+'/'+filename)    
    return image

def load_labels_dagm(path,class_number,num,exp=''):
    """
    Input: 
    
    Output: 
    
    """
    i = num - 1
    pathF = path+str(class_number)+'_def'
    f = open(pathF+'/'+'labels.txt', 'r')
    lines = f.readlines()
    gt = ground_truth_defectA(lines,i)
    return gt
    
def load_list_selected_images(defect,class_number,number_experimet):
    if defect == '':
        dst = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_'+str(number_experimet)+'_DAGM/Class'+str(class_number)+'/Cl_'+str(class_number)+'_NO/'
    elif defect == 'A':
        dst = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_'+str(number_experimet)+'_DAGM/Class'+str(class_number)+'/Cl_'+str(class_number)+'_A/'
    else:
        dst = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_'+str(number_experimet)+'_DAGM/Class'+str(class_number)+'/Cl_'+str(class_number)+'_'+str(defect)+'/'
    if defect == '':
        list_name = 'list_no_defect.pckl'
    elif defect == 'A':
        list_name = 'list_defect_A.pckl'
    elif defect == 'AB':
        list_name = 'list_defect_AB.pckl'
    elif defect == 'B':
        list_name = 'list_defect_B.pckl'
    f = open(dst+list_name, 'rb')
    lista = pickle.load(f)
    f.close()
    return lista
    
def move_selected_images(lista,defect,class_number,number_experimet):
    path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'
    if defect =='':
        src = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(class_number)+'/'
        addon = ''
    elif defect == 'A':
        src = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(class_number)+'_def/'
        addon = ''
    else:
        src = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(class_number)+'_def'+defect+'/'
        addon = '_'
    if defect == '':
        dst = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_'+str(number_experimet)+'_DAGM/Class'+str(class_number)+'/Cl_'+str(class_number)+'_NO/'
    else:
        dst = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_'+str(number_experimet)+'_DAGM/Class'+str(class_number)+'/Cl_'+str(class_number)+'_'+str(defect)+'/'
    file_type = '.png'
    contador = 1
    for i in range (0,len(lista)):
        print i
        if defect == '':
            filename = str(lista[i])+file_type
        elif defect == 'A':
            filename = str(lista[i])+file_type
        else :
            filename = str(lista[i])+addon+defect+file_type
        filename_dest = 'Cl_'+ str(class_number)+'_'+str(i+1)+'_'+defect+file_type
        copyfile(src+filename, dst+filename_dest)
        if defect == 'A': 
            cord_defect_A = get_labels_defectA(path,class_number,int(lista[i]))
            cord_roi = get_roi_rect(path,class_number,int(lista[i]))
            write_labels_defectA(class_number,lista[i],cord_defect_A,reason='new_experiment',new_num=contador,defect=defect)
            write_labels_expROI(class_number,lista[i],cord_roi['x1'],cord_roi['y1'],cord_roi['x2'],cord_roi['y2'],reason='new_experiment',new_num=contador,defect=defect)
        elif defect == 'B':
             cord_defect_B = get_labels_defectB(path,class_number,int(lista[i]))
             cord_roi = get_roi_rect(path,class_number,int(lista[i]))
             write_labels_defectB(class_number,lista[i],cord_defect_B['x1'],cord_defect_B['y1'],cord_defect_B['x2'],cord_defect_B['y2'],reason='new_experiment',new_num=contador,defect=defect)
             write_labels_expROI(class_number,lista[i],cord_roi['x1'],cord_roi['y1'],cord_roi['x2'],cord_roi['y2'],reason='new_experiment',new_num=contador,defect=defect)
        elif defect == 'AB': 
            cord_defect_A = get_labels_defectA(path,class_number,int(lista[i]))
            cord_defect_B = get_labels_defectB(path,class_number,int(lista[i]))
            cord_roi = get_roi_rect(path,class_number,int(lista[i]))
            write_labels_defectA(class_number,lista[i],cord_defect_A,reason='new_experiment',new_num=contador,defect=defect)
            write_labels_defectB(class_number,lista[i],cord_defect_B['x1'],cord_defect_B['y1'],cord_defect_B['x2'],cord_defect_B['y2'],reason='new_experiment',new_num=contador,defect=defect)
            write_labels_expROI(class_number,lista[i],cord_roi['x1'],cord_roi['y1'],cord_roi['x2'],cord_roi['y2'],reason='new_experiment',new_num=contador,defect=defect)
        else:
            print ('No defect')
        contador+=1
    return 1
    
def rectangle_expanded_roi(gt):
    """
    Input: 
    
    Output: 
    
    """
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
    
def save_image_defect(defect,num,cl_number,image):
    filename_with_defect = str(num) + '_'+defect+'.png'
    cv2.imwrite('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(cl_number)+"_def"+defect+"/"+filename_with_defect, image)
     
def save_list_selected_images(lista,defect,class_number,number_experimet):
    if defect == '':
        dst = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_'+str(number_experimet)+'_DAGM/Class'+str(class_number)+'/Cl_'+str(class_number)+'_NO/'
    elif defect == 'A':
        dst = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_'+str(number_experimet)+'_DAGM/Class'+str(class_number)+'/Cl_'+str(class_number)+'_A/'
    else:
        dst = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_'+str(number_experimet)+'_DAGM/Class'+str(class_number)+'/Cl_'+str(class_number)+'_'+str(defect)+'/'
    if defect == '':
        list_name = 'list_no_defect.pckl'
    elif defect == 'A':
        list_name = 'list_defect_A.pckl'
    elif defect == 'AB':
        list_name = 'list_defect_AB.pckl'
    elif defect == 'B':
        list_name = 'list_defect_B.pckl'
    f = open(dst+list_name, 'wb')
    pickle.dump(lista, f)
    f.close()
    return 1
    
def select_images(defect,class_number):
    """
    Input: Defect type and class number. 
    Output: Array with number of images to select.    
    """
    list_images = []
    file_type = '.png'
    if defect =='':
        src = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(class_number)+'/'
    elif defect == 'A':
        src = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(class_number)+'_def/'
    else:
        src = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(class_number)+'_def'+defect+'/'

    for i in range (1,151):
        if defect =='':
            filename = str(i)+file_type
        elif defect == 'A':
            filename = str(i)+file_type
        else:
            filename = str(i)+'_'+defect+file_type
        image = cv2.imread(src+filename) 
        cv2.imshow('Image',image)
        var = raw_input("Do you want to use this Image?: ")
        if var == '1': 
            print ('Image '+str(i)+' added')
            list_images.append(i)
        else:
            print ('Image '+str(i)+' discarded ')
        cv2.destroyAllWindows()
        if len(list_images)>=100:
            break
    return list_images

def write_labels_defectA(cl_number,num,gt,reason='',new_num='',defect=''):
    """
    Input: 
    
    Output: 
    Nota: ahora mismo esta quemada rutas y parametros [Modificar]
    """
    if reason == 'new_experiment':
        if defect=='A':
            target = open('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'+str(cl_number)+'/Cl_'+str(cl_number)+'_A/'+'labels_defect_A.txt', 'a')
            line = str(new_num)+'\t'+ str(gt['semi_major_ax'])+'\t'+ str(gt['semi_minor_ax'])+'\t'+ str(int(gt['rotation_angle']))+'\t'+ str(gt['x_position_center'])+'\t'+ str(gt['y_position_center'])
        else:
            target = open('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'+str(cl_number)+'/Cl_'+str(cl_number)+'_'+str(defect)+'/'+'labels_defect_A.txt', 'a')
            line = str(new_num)+'\t'+ str(gt['semi_major_ax'])+'\t'+ str(gt['semi_minor_ax'])+'\t'+ str(int(gt['rotation_angle']))+'\t'+ str(gt['x_position_center'])+'\t'+ str(gt['y_position_center'])
    else:
        target = open('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(cl_number)+'_defAB/'+'labels_defect_A.txt', 'a')
        line = str(num)+'\t'+ str(gt['semi_major_ax'])+'\t'+ str(gt['semi_minor_ax'])+'\t'+ str(int(gt['rotation_angle']))+'\t'+ str(gt['x_position_center'])+'\t'+ str(gt['y_position_center'])
    target.write(line)
    target.write("\n")
    target.close()
    return True
    
def write_labels_defectB(cl_number,num,x1,y1,x2,y2,reason='',new_num='',defect=''):
    """
    Input: 
    
    Output: 
    Nota: ahora mismo esta quemada rutas y parametros [Modificar]
    """
    if reason == 'new_experiment':
        target = open('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'+str(cl_number)+'/Cl_'+str(cl_number)+'_'+str(defect)+'/'+'labels_defect_B.txt', 'a')
        line = str(new_num)+'\t'+str(x1)+'\t'+str(y1)+'\t'+str(x2)+'\t'+str(y2)
    else:
        target = open('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(cl_number)+'_def'+defect+'/'+'labels_defect_B.txt', 'a')
        line = str(num)+'\t'+str(x1)+'\t'+str(y1)+'\t'+str(x2)+'\t'+str(y2)
    target.write(line)
    target.write("\n")
    target.close()
    return True
    
def write_labels_expROI(cl_number,num,x1,y1,x2,y2,reason='',new_num='',defect=''):
    """
    Input:
    Output: 
    Nota: ahora mismo esta quemada rutas y parametros [Modificar]
    """
    if reason == 'new_experiment':
        target = open('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'+str(cl_number)+'/Cl_'+str(cl_number)+'_'+str(defect)+'/'+'ROI.txt', 'a')
        line = str(new_num)+'\t'+str(x1)+'\t'+str(y1)+'\t'+str(x2)+'\t'+str(y2)
    else:
        target = open('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(cl_number)+'_def'+defect+'/'+'ROI.txt', 'a')
        line = str(num)+'\t'+str(x1)+'\t'+str(y1)+'\t'+str(x2)+'\t'+str(y2)
    target.write(line)
    target.write("\n")
    target.close()
    return True