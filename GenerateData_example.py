# -*- coding: utf-8 -*-
"""
@author: josemiguelarrieta
"""

#Image Manipulation
from __future__ import print_function
import os
os.chdir('Documents/SIVA')
import cv2
from utils_dagm import WeakLabeling, get_data_SISL, get_data_SIML, get_data_MISL, get_data_MIML

#Image Information [1 Image Only]
class_number = 1
number_experimet = 1
num = 1
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'

#WeakLabeling defect A
cropped,cropped_mask = WeakLabeling(path,num,class_number,defect = 'A',exp = True)

cv2.imshow("cropped", cropped)
cv2.imshow("cropped_mask", cropped_mask)

#WeakLabeling defect AB [MULTILABEL]
cropped,cropped_maskA,cropped_maskB = WeakLabeling(path,num,class_number,defect = 'AB',exp = True)

cv2.imshow("cropped", cropped)
cv2.imshow("cropped_maskA", cropped_maskA)
cv2.imshow("cropped_maskB", cropped_maskB)
cv2.destroyAllWindows()

#SISL [1 imagen]
labels_NO,instances_NO = get_data_SISL(cropped) #No defect
labels_A,instances_A = get_data_SISL(cropped,cropped_maskA) #defectA
labels_B,instances_B = get_data_SISL(cropped,cropped_maskB) #defectB
labels_AB,instances_AB = get_data_SISL(cropped,cropped_maskA,cropped_maskB) #defectAB
                
#MISL[1 imagen]
label_bag, bag = get_data_MISL(cropped)
label_bagA, bagA = get_data_MISL(cropped,cropped_maskA)
label_bagB, bagB = get_data_MISL(cropped,cropped_maskB)
label_bagAB, bagAB = get_data_MISL(cropped,cropped_maskA,cropped_maskB)
         
#SIML[1 imagen]
insta_labels_NO, instancesNO = get_data_SIML(cropped,cropped_maskA=None,cropped_maskB=None)
insta_labels_A, instances_A = get_data_SIML(cropped,cropped_maskA,cropped_maskB=None)
insta_labelsB, instancesB = get_data_SIML(cropped,cropped_maskA=None,cropped_maskB=cropped_maskB)
insta_labels_AB, instances_AB = get_data_SIML(cropped,cropped_maskA,cropped_maskB)

#MIML[1 imagen]  
labels_bagA, bagA = get_data_MIML(cropped,cropped_maskA=None,cropped_maskB=None)          
labels_bagA, bagA = get_data_MIML(cropped,cropped_maskA=cropped_maskA,cropped_maskB=None)
labels_bagB, bagB = get_data_MIML(cropped,cropped_maskA=None,cropped_maskB=cropped_maskB)
labels_bagAB, bagAB = get_data_MIML(cropped,cropped_maskA=cropped_maskA,cropped_maskB=cropped_maskB)
