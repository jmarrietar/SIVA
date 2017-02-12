# -*- coding: utf-8 -*-
"""
@author: josemiguelarrieta
"""

#Image Manipulation
from __future__ import print_function
import os
os.chdir('Documents/SIVA')
import cv2
import numpy as np
from utils_dagm import WeakLabeling, get_data_SISL, get_data_SIML, get_data_MISL, get_data_MIML

#Image Information [1 Image Only]
class_number = 1
number_experimet = 1
num = 14
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'

#WeakLabeling

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
labels,instances = get_data_SISL(cropped,cropped_mask)
                
#MISL[1 imagen]
label_bag, bag = get_data_MISL(cropped,cropped_mask)
         
#SIML[1 imagen]
insta_labels, instances = get_data_SIML(cropped,cropped_maskA,cropped_maskB)
       
#MIML[1 imagen]
labels_bag, bag = get_data_MIML(cropped,cropped_maskA,cropped_maskB)