#Image Manipulation
from __future__ import print_function
import os
os.chdir('Documents/SIVA')
import numpy as np
from utils_dagm import WeakLabeling, get_data_SISL, get_data_SIML, get_data_MISL, get_data_MIML

#Path
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'
#Class Number
class_number = 1
#Experiment Number
number_experimet = 1

#Defect Type
defect = 'A'
defect = 'B'
defect = 'AB'
defect = 'No'

#------#
#-SISL-#
#------#
X =  np.empty((0,58), float)    #58 is the number of features [Change]
Y = np.empty((0,1), int)

for i in range (1,100):
    #Image number 
    num = i
    #WeakLabeling
    cropped,cropped_mask = WeakLabeling(path,num,class_number,defect = defect,exp = True)
    labels,instances = get_data_SISL(cropped,cropped_mask)
    X = np.concatenate((X, instances), axis=0)
    Y = np.concatenate((Y, labels), axis=0)

#Save Data

#Load Data

#------#
#-MISL-#
#------#
Bags = []
labels_bags = np.empty((0,1), int)

for i in range (1,100):
    #Image number 
    num = i
    #WeakLabeling
    cropped,cropped_mask = WeakLabeling(path,num,class_number,defect = defect,exp = True)
    label,bag = get_data_MISL(cropped,cropped_mask)
    Bags.append(bag)
    labels_bags = np.concatenate((labels_bags, np.array([label]).reshape(1,1)), axis=0)

#Save Data

#Load Data

#------#
#-SIML-#
#------#
X =  np.empty((0,58), float)    #58 is the number of features [Change]
Y = np.empty((0,2), int)

for i in range (1,100):
    #Image number 
    num = i
    #WeakLabeling
    cropped,cropped_maskA,cropped_maskB = WeakLabeling(path,num,class_number,defect = 'AB',exp = True)
    insta_labels, instances = get_data_SIML(cropped,cropped_maskA,cropped_maskB)
    X = np.concatenate((X, instances), axis=0)
    Y = np.concatenate((Y, insta_labels), axis=0)

#Save Data

#Load Data

#------#
#-MIML-#
#------#
defect = 'No'
defect = 'AB'

Bags = []
labels_bags = np.empty((0,2), int)
for i in range (1,100):
    #Image number 
    num = i
    #WeakLabeling
    cropped,cropped_maskA,cropped_maskB = WeakLabeling(path,num,class_number,defect = defect,exp = True)
    labels_bag, bag = get_data_MIML(cropped,cropped_maskA,cropped_maskB)
    Bags.append(bag)
    labels_bags = np.concatenate((labels_bags, np.array([labels_bag]).reshape(1,2)), axis=0)
    
#Save Data

#Load Data