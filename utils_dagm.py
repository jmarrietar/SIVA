# -*- coding: utf-8 -*-
"""
Utilities for DAGM  images experimentation. 

@author: josemiguelarrieta
"""
import cv2
import numpy as np
import math
import pickle
from shutil import copyfile
from skimage.feature import hog
from skimage import color
from localbinarypatterns import LocalBinaryPatterns
from balu.ImageProcessing import Bim_segbalu
from Bfx_basicint import Bfx_basicint
from skimage.feature import hog
import mahotas.features

def add_salt(image,cl_number):
    """
    Add salt to an Image.
        
    Input data:
        image = image to add salt.
        cl_number = class number of image.
    Output:
        Image with salt.
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
    Blur and Image
        
    Input data:
        image2blur = image to be blured.
        cl_number = class number of image.
        
    Output:
        Image blurred.
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
    Add defect type B(Blured) to image
    
    Input:
        c11, c22, A, B: cordinates where defect should be placed
        blured: image blured.
        image : image to add defect.
    
    Output: 
        image = image with defect B
    
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
    
def Defect(X,Y,window_mask):
    """
    Input:
    
    Output: 
    
    """
    if (float(np.count_nonzero(window_mask))/(X*Y)>0.10):
        return True
    else:
        return False

def defect_B_rect_ROI(x1, x, y1, y2):
    """
    Roi rectangle delimitation creation for defect B.
    
    Input: 
        x1, x, y1, y2 : Big Roi coordinates on delimitation for both defects.
    
    Output:
        d1, d2,  d3, d4 : Coordinates for little Roi Delimitation on defect B.
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
    Return Coordinates of Ellipse inside given rectangle.
    
    Input: 
        x1,y1,x,d4 : Rectangle Coordinates
    
    Output: 
        c11, c22, A, B : Ellipse Coordinates (No rotation)
    
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
    
def extract_features (image):
    """
    Extract Features. 
        -> Features Extracted: 
            * LBP
            * Bfx_Basicint
            * Haralick
            * free Threshold Adjacency Statistics
            * Zernike Moments
            * HOG

    """    
    gray_image = color.rgb2gray(image)
    
    #LBP Features
    desc = LocalBinaryPatterns(12, 4)
    lbp_hist = desc.describe(gray_image)
    lbp_hist = lbp_hist.reshape(1,len(lbp_hist))

    #Bfx_Basicint
    R,_,_ = Bim_segbalu(gray_image)
    options = {'show': True, 'mask': 5}
    basicint, Xn = Bfx_basicint(gray_image,R,options)

    #Haralick features
    haralick = mahotas.features.haralick(gray_image).mean(0)
    haralick = haralick.reshape(1,len(haralick))

    #parameter free Threshold Adjacency Statistics
    pftas = mahotas.features.pftas(gray_image)
    pftas = pftas.reshape(1,len(pftas))

    #Zernike Moments
    zernike = mahotas.features.zernike_moments(gray_image, radius=2)
    zernike = zernike.reshape(1,len(zernike))

    #HOG [Fix Dimentionality]
    HOG = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=False)
    HOG = HOG.reshape(1,len(HOG))

    #Join Features
    features = np.concatenate((lbp_hist,basicint,haralick,pftas,zernike,HOG), axis=1)
    
    return features
    
def get_coordinates_crop(x1,x2,y1,y2,length,width): 
    """
    Get coordinates to crop ROI in image, useful when ROI es greater than
    image dimentions. 
        
    Input:
    x1,x2,y1,y2:  Coordinates of ROI in image.
    length,width: Dimentions of image
    
    Output: 
    start_x, end_x, start_y, end_y:  Dimentions of ROI cropped to image. 
    """
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
        start_y = length
    else:
        start_y = y1
    if (y2 < 0):
        end_y = 0
    elif (y2 > width):   
        end_y = width
    else:
        end_y = y2        
    return start_x, end_x, start_y, end_y
    
def get_data_SISL(cropped,cropped_maskA=None,cropped_maskB=None):
    """ 
    Labeling & Feature Extracion     
    
    Input:
        cropped = Cropped Section of Image. 
        cropped_maskA = Cropped Mask For Defect.
        cropped_maskB = Cropped Mask For Defect 2.
    Output: 
        If defect is AB return generalization of defect and respective defects as one. 
        Otherwise if defect is 1 or 2(One Defect) return instances and labels accordingly. 
    
    TO DO: Change number of Feature from 58 to X
    """
    (winW, winH) = (32, 32)                 #Descomposición en imágenes de 32 a 32 con corrimiento de a 32?. 
    labels = np.empty((0,1), int)
    instances =  np.empty((0,58), float)    #Cambiar 58 por numero de Features
    insta_labelsA = np.empty((0,1), int)
    insta_labelsB = np.empty((0,1), int)
            
    #Check If image contains Defect 2
    if cropped_maskB is None:
        cropped_maskB = np.zeros(cropped.shape[:2], dtype = "uint8")
       
    #Check If image contains Defect 1
    if cropped_maskA is None:
        cropped_maskA = np.zeros(cropped.shape[:2], dtype = "uint8")
            
    #If image contains Defect 2 Record labels of defect
    if cropped_maskB is not None:
        insta_labelsB = np.empty((0,1), int)
        for (x, y, window_maskB,window) in sliding_window(cropped_maskB,cropped, stepSize=32, windowSize=(winW, winH)):
            #if the window does not meet our desired window size, ignore it
            if window_maskB.shape[0] != winH or window_maskB.shape[1] != winW:
                continue
            #Label defectB
            if (Defect(winW,winH,window_maskB)==True):
                insta_labelsB = np.append(insta_labelsB,np.array([[1]]), axis = 0)
            else:
                insta_labelsB = np.append(insta_labelsB,np.array([[0]]), axis = 0)
                
    #Do labeling and feature extraction         
    for (x, y, window_mask,window) in sliding_window(cropped_maskA,cropped, stepSize=32, windowSize=(winW, winH)):
        #if the window does not meet our desired window size, ignore it
        if window_mask.shape[0] != winH or window_mask.shape[1] != winW:
            continue
        if (Defect(winW,winH,window_mask)==True):
            insta_labelsA = np.append(insta_labelsA,np.array([[1]]), axis = 0)
        else:
            insta_labelsA = np.append(insta_labelsA,np.array([[0]]), axis = 0)
        #Extract Features
        instance = extract_features(window) 
        instance.resize(1,len(instance))
        instances = np.append(instances,instance,axis=0)
        
    #If cropped_mask1 & cropped_mask2 exist then defect is AB[Generalization]
    if 255 in cropped_maskA  and 255 in cropped_maskB:
        for i in range(0,len(insta_labelsB)):
            if (insta_labelsA[i][0]==1 or insta_labelsB[i][0]==1):
                labels = np.append(labels,np.array([[1]]), axis = 0)
            else:
                labels = np.append(labels,np.array([[0]]), axis = 0)
        return labels,instances
    else:
        return insta_labelsA, instances
        
def get_data_MISL(cropped,cropped_maskA = None,cropped_maskB = None):
    """        
    Labeling & Feature Extracion     
    
    Input:
        cropped = Cropped Section of Image. 
        cropped_maskA = Cropped Mask For Defect.
        cropped_maskB = Cropped Mask For Defect 2 (default:None).
    Output: 
        If defect is AB return generalization of defect and respective defects as one. 
        Otherwise if defect is A or B(One Defect) return instances and labels accordingly.
    
    """
    if cropped_maskA is not None and cropped_maskB is not None: #Two Defect
        labels,bag = get_data_SISL(cropped,cropped_maskA,cropped_maskB)
        if 1 in labels:
            label_bag = 1
        else:
            label_bag = 0
        return label_bag, bag
    elif cropped_maskA is not None:   #One Defect
        labels,bag = get_data_SISL(cropped,cropped_maskA)
        if 1 in labels:
            label_bag = 1
        else:
            label_bag = 0
        return label_bag, bag
    elif cropped_maskA is None and cropped_maskB is None:  #No Defect
        labels,bag = get_data_SISL(cropped)
        return 0, bag

def get_data_SIML(cropped,cropped_maskA,cropped_maskB):
    """
    With SIML you can extract all combination from instance in Image with AB {0,1}{0,0}{1,0}{1,1}
    
    Input:
    
    Output: 
    
    Labeling & Feature Extracion 
    
    
    """
    
    instances =  np.empty((0,58), float)    #Cambiar 58 por numero de Features
    insta_labelsA = np.empty((0,1), int)
    insta_labelsB = np.empty((0,1), int)
    (winW, winH) = (32, 32)                 #Descomposición en imágenes de 32 a 32 con corrimiento de a 32?. 
    
    #If defect is None
    if cropped_maskA is None: 
        cropped_maskA = np.zeros(cropped.shape[:2], dtype = "uint8")
    #If is just one defect
    if cropped_maskB is None: 
        cropped_maskB = np.zeros(cropped.shape[:2], dtype = "uint8")

    for (x, y, window_maskB,window) in sliding_window(cropped_maskB,cropped, stepSize=32, windowSize=(winW, winH)):
        #if the window does not meet our desired window size, ignore it
        if window_maskB.shape[0] != winH or window_maskB.shape[1] != winW:
            continue
        #Label defectA
        if (Defect(winW,winH,window_maskB)==True):
            insta_labelsB = np.append(insta_labelsB,np.array([[1]]), axis = 0)
        else:
            insta_labelsB = np.append(insta_labelsB,np.array([[0]]), axis = 0)
            
    for (x, y, window_maskA,window) in sliding_window(cropped_maskA,cropped, stepSize=32, windowSize=(winW, winH)):
        #if the window does not meet our desired window size, ignore it
        if window_maskA.shape[0] != winH or window_maskA.shape[1] != winW:
            continue     
        #Label defectB
        if (Defect(winW,winH,window_maskA)==True):
            insta_labelsA = np.append(insta_labelsA,np.array([[1]]), axis = 0)
        else:
            insta_labelsA = np.append(insta_labelsA,np.array([[0]]), axis = 0)
        #Extract Features        
        instance = extract_features(window)
        instance.resize(1,len(instance))
        instances = np.append(instances,instance,axis=0)
    insta_labels = np.concatenate((insta_labelsA,insta_labelsB),axis=1)        
    return insta_labels, instances

def get_data_MIML(cropped,cropped_maskA,cropped_maskB):
    """
    
    With MIML is necesary images of defects AB, A, B and None
    
    Input:
    
    Output:
    
    """
    insta_labels, instances = get_data_SIML(cropped,cropped_maskA,cropped_maskB)
    insta_labelsA = insta_labels[:,[0]]
    insta_labelsB = insta_labels[:,[1]]
    bag = instances
    if 1 in insta_labelsA:
        labelA = 1
    else:
        labelA = 0
    if 1 in insta_labelsB:
        labelB = 1
    else:
        labelB = 0
    label_bag = np.array([[labelA,labelB]])
    return label_bag, bag
        
    
def get_labels_defectA(path,class_number,num,exp='',defect='',degrees=False):
    """
    Input:
        path = Path where labels are located.
        class_number = class number
        num = number of image
        exp = check if is for an experiment or not.
        defect = Type of defect.
        degrees = check if necessary to convert to degrees.
    
    Output:
        gt : Dictionary with ground truth for defect A.
        
    Note: Paths are fixed.(Fix)
    """
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
        path = Path where labels are located.
        class_number = class number
        num = number of image
        exp = check if is for an experiment or not.
        defect = Type of defect.
        degrees = check if degrees already on.
    
    Output:
        gt =  Dictionary with ground truth for defect B.
        
    Note: Paths are fixed.(Fix)
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
    """
    Get Roi Coordinates of defects
    
    Input:
        path = Path where labels are located.
        class_number = class number
        num = number of image
        exp = check if is for an experiment or not.
        defect = Type of defect.
        
        Output:
        
        Note: Paths are fixed.(Fix)
    """
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
        lines = All lines on file .txt
        line_number = specific line on lines to read.
    Output:
        dictionary with labels
    
    Note: Paths are fixed.(Fix)
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
        lines = All lines on file .txt
        line_number = specific line on lines to read.
        degrees = check if degrees already on.
    
    Output:
        dictionary with labels
    
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
        path = Path where images located.
        num_image = number of image
        class_number = class number
        defect = defect type
        exp = if it is for experiment
    
    Output: 
        image = Returned image
    """
    
    if exp == True:
        pathF = path+str(class_number)+'/Cl_'+str(class_number)+'_'+defect
        if defect =='NO':
            defect = ''
        filename = 'Cl_'+str(class_number)+'_'+str(num_image)+'_'+defect+'.png'
        image = cv2.imread(pathF+'/'+filename)
    else:
        filename = str(num_image)+'.png'
        pathF = path+str(class_number)+defect
        image = cv2.imread(pathF+'/'+filename)    
    return image

def load_labels_dagm(path,class_number,num,exp=''):
    """
    Alternative function to load labels A.
    
    see get_labels_defectA
    """
    i = num - 1
    pathF = path+str(class_number)+'_def'
    f = open(pathF+'/'+'labels.txt', 'r')
    lines = f.readlines()
    gt = ground_truth_defectA(lines,i)
    return gt
    
def load_list_selected_images(defect,class_number,number_experimet):
    """
    Input:
        defect = defect type
        class_number = Class number
        number_experimet = Number of experiment
        
    Output:
        lista: Lista of selected images
    
    """
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
    """
    Input:
        lista = Lista of selected images
        defect = Type of defect.
        class_number = Class number
        number_experimet = Number of experiment
    
    Output:
        return 1 when finish.
    
    Note: Paths are fixed.(Fix)
    """
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
    Create a rectangle ROI based on ellypse coordinates.
        
    Input: 
        gt: Ground truth of Ellipse Coordinates.
    
    Output: 
        x1, y1, x2, y2, x, y Coordinates of ROI
    
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
    """
    Save Image with defect.
    
    Input:
        defect = Type of defect
        num = number of image
        cl_number = Class number
        image = image to be saved.
        
    Note: Paths are fixed.(Fix)
    """
    filename_with_defect = str(num) + '_'+defect+'.png'
    cv2.imwrite('/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(cl_number)+"_def"+defect+"/"+filename_with_defect, image)
     
def save_list_selected_images(lista,defect,class_number,number_experimet):
    """
    Save lista for defect type.
        
    Note: Paths are fixed.(Fix)
    """
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
    Load images from class number to user select witch ones to use in the experiments.
    
    Input: 
        defect = Defect type
        class_number = class number.
    Output: 
        list_images = Array with number of images to select.
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
    
def sliding_window(image, image2 ,stepSize, windowSize):
    """
    Sliding windows implementation.    
    """
    # slide a window across the image
    for y in xrange(0, image2.shape[0], stepSize):
        for x in xrange(0, image2.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]],image2[y:y + windowSize[1], x:x + windowSize[0]])


def write_labels_defectA(cl_number,num,gt,reason='',new_num='',defect=''):
    """
    Input:
        cl_number = Class number
        num = Number of Image
        gt = Ground truth Labels A
        reason= if is for experiment
        new_num = new number of Image
        defect = Defect Type.
    
    Output: 
        Return true
    
    Note: Paths are fixed.(Fix)
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
        cl_number = Class number
        num = Number of Image
        x1,y1,x2,y2 = Coordinates defect B Roi
        reason = if it is for experiment
        new_num = new number of Image
        defect = defect Type
    
    Output: 
        Return true
    
    Note: Paths are fixed.(Fix)
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
        cl_number = Class number
        num = Number of Image
        x1,y1,x2,y2 = Coordinates of ROI
        reason= if it is for experiment
        new_num= new number of Image
        defect= defect Type
    
    Output: 
        Return true
    
    Note: Paths are fixed.(Fix)
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
    
def WeakLabeling(path,num,class_number,defect = 'X',exp = True):
    """    
    This function was implemented for using WeekLabeling. 
    
    Input:
        path: Path Where images are located.
        num :Number of Image.
        class_number: Class number of image.
        defect: Type of defect.
        exp: if it is a experiment (default: True)
    
    Output: 
        roi: ROI cropped. 
        roi_mask: Mask of ROI cropped.
    """
    image = load_image_dagm(path,num,class_number,defect = defect,exp = True)
    length, width,_ = image.shape
    #Create Mask of Image Uploaded!.
    mask = np.zeros(image.shape[:2], dtype = "uint8") 
    #get roi
    roi_labels = get_roi_rect(path,class_number,num,exp=True,defect=defect)
    if defect == 'A': 
        #get labels defectA
        dfA_A = get_labels_defectA(path,class_number,num,exp=True,defect='A',degrees=True)    
        #Draw Defect on Mask
        cv2.ellipse(mask,(dfA_A['x_position_center'],dfA_A['y_position_center']),(dfA_A['semi_major_ax'],dfA_A['semi_minor_ax']),dfA_A['rotation_angle'],0,360,255,-1)  #Draw Ellipse [Ground Truth]
    elif defect == 'B':
        #get labels defectBs
        dfB_B = get_labels_defectB(path,class_number,num,exp=True,defect='B')
        #Draw Defect on Mask
        cv2.rectangle(mask,(dfB_B['x1'],dfB_B['y1']),(dfB_B['x2'],dfB_B['y2']),255,-1)
    elif defect == 'AB':
        dfAB_A = get_labels_defectA(path,class_number,num,exp=True,defect='AB',degrees=True)
        dfAB_B = get_labels_defectB(path,class_number,num,exp=True,defect='AB')
        maskA = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(maskA,(dfAB_A['x_position_center'],dfAB_A['y_position_center']),(dfAB_A['semi_major_ax'],dfAB_A['semi_minor_ax']),dfAB_A['rotation_angle'],0,360,255,-1)
        maskB = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.rectangle(maskB,(dfAB_B['x1'],dfAB_B['y1']),(dfAB_B['x2'],dfAB_B['y2']),255,-1)
        #Week labeling [Cropping-ROI]        
        start_x, end_x, start_y, end_y = get_coordinates_crop(roi_labels['x1'],roi_labels['x2'],roi_labels['y1'],roi_labels['y2'],length,width)
        roi = image[start_y:end_y,start_x:end_x]      
        roi_maskA = maskA[start_y:end_y,start_x:end_x]
        roi_maskB = maskB[start_y:end_y,start_x:end_x]
        return roi,roi_maskA,roi_maskB
    else:
        return 0,0
    #Week labeling [Cropping-ROI]
    start_x, end_x, start_y, end_y = get_coordinates_crop(roi_labels['x1'],roi_labels['x2'],roi_labels['y1'],roi_labels['y2'],length,width)
    roi = image[start_y:end_y,start_x:end_x]      
    roi_mask = mask[start_y:end_y,start_x:end_x]
    return roi,roi_mask