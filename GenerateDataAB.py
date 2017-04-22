#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
import pickle
from utils_dagm import WeakLabeling,load_image_dagm
from LabelExtractFeatures import get_data_SISL, get_data_SIML, get_data_MISL, get_data_MIML

#Path
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'
ClassNumber = sys.argv[1]
defects =['AB','NO']

"""
Generate Data for Defect AB with Single Instance Single Learning
"""
                                        #------#
                                        #-SISL-#
                                        #------#
InstanceType = 'SISL'

X_sisl =  np.empty((0,144), float)    #144 is the number of Features
Y_sisl = np.empty((0,1), int)

for defect in defects:
    for i in range (1,100):
        num = i #Image number 
        if defect == 'AB':
            cropped,cropped_maskA,cropped_maskB = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            labels,instances = get_data_SISL(cropped,cropped_maskA,cropped_maskB)
        elif defect == 'NO':
            image = load_image_dagm(path,num,ClassNumber,defect = defect,exp = True)
            labels,instances = get_data_SISL(image)
        else:
            cropped,cropped_mask = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            labels,instances = get_data_SISL(cropped,cropped_mask)
        X_sisl = np.concatenate((X_sisl, instances), axis=0)
        Y_sisl = np.concatenate((Y_sisl, labels), axis=0)

#Save Data
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'X_sisl.pckl', 'wb')
pickle.dump(X_sisl, f)
f.close()
f = open(path_data+'Y_sisl.pckl', 'wb')
pickle.dump(Y_sisl, f)
f.close()
print("SISL data saved.")


"""
Generate Data for Defect AB with Multi-Instance Single Learning
"""
                                        #------#
                                        #-MISL-#
                                        #------#
InstanceType = 'MISL'

Bags_misl = []
Y_misl = np.empty((0,1), int)

for defect in defects:
    for i in range (1,100):
        num = i #Image number 
        if defect == 'AB':
            cropped,cropped_maskA,cropped_maskB = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            labels_bags, bag = get_data_MISL(cropped,cropped_maskA,cropped_maskB)
        elif defect == 'NO':
            image = load_image_dagm(path,num,ClassNumber,defect = defect,exp = True)
            labels_bags, bag = get_data_MISL(image)
        else:
            cropped,cropped_mask = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            labels_bags,bag = get_data_MISL(cropped,cropped_mask)
        Bags_misl.append(bag)
        Y_misl = np.concatenate((Y_misl, np.array([labels_bags]).reshape(1,1)), axis=0)

#Save Data
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'Bags_misl.pckl', 'wb')
pickle.dump(Bags_misl, f)
f.close()
f = open(path_data+'Y_misl.pckl', 'wb')
pickle.dump(Y_misl, f)
f.close()
print("MISL data saved.")


"""
Generate Data for Defect AB with Single Instance Multi Learning
"""
                                        #------#
                                        #-SIML-#
                                        #------#
InstanceType = 'SIML'

X_siml =  np.empty((0,144), float)    #58 is the number of features [Change]
Y_siml = np.empty((0,2), int)

for defect in defects:
    for i in range (1,100):
        num = i     #Image number 
        if defect == 'AB':
            cropped,cropped_maskA,cropped_maskB = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            labels, instances = get_data_SIML(cropped,cropped_maskA,cropped_maskB)
        elif defect == 'NO':
            image = load_image_dagm(path,num,ClassNumber,defect = defect,exp = True)
            labels, instances = get_data_SIML(image,cropped_maskA=None,cropped_maskB=None)
        elif defect == 'A':
            cropped,cropped_maskA = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            labels, instances = get_data_SIML(cropped,cropped_maskA,cropped_maskB=None)
        elif defect =='B':
            cropped,cropped_maskB = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            labels, instances = get_data_SIML(cropped,cropped_maskA=None,cropped_maskB=cropped_maskB)
        X_siml = np.concatenate((X_siml, instances), axis=0)
        Y_siml = np.concatenate((Y_siml, labels), axis=0)

#Save Data
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'X_siml.pckl', 'wb')
pickle.dump(X_siml, f)
f.close()
f = open(path_data+'Y_siml.pckl', 'wb')
pickle.dump(Y_siml, f)
f.close()
print("SIML data saved.")

"""
Generate Data for Defect AB with Multi Instance Multi Learning
"""
                                        #------#
                                        #-MIML-#
                                        #------#
InstanceType = 'MIML'

Bags_miml = []
Y_miml = np.empty((0,2), int)

for defect in defects:
    for i in range (1,100):
        num = i     #Image number 
        if defect == 'AB':    
            cropped,cropped_maskA,cropped_maskB = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            labels_bag, bag = get_data_MIML(cropped,cropped_maskA,cropped_maskB)
        elif defect == 'NO':
            image = load_image_dagm(path,num,ClassNumber,defect = defect,exp = True)
            labels_bag, bag = get_data_MIML(image,cropped_maskA=None,cropped_maskB=None)  
        elif defect == 'A':
            cropped,cropped_maskA = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            labels_bag, bag = get_data_MIML(cropped,cropped_maskA=cropped_maskA,cropped_maskB=None)
        elif defect == 'B':
            cropped,cropped_maskB = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            labels_bag, bag = get_data_MIML(cropped,cropped_maskA=None,cropped_maskB=cropped_maskB)
        Bags_miml.append(bag)
        Y_miml = np.concatenate((Y_miml, labels_bag), axis=0)
    
#Save Data
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'Bags_miml.pckl', 'wb')
pickle.dump(Bags_miml, f)
f.close()
f = open(path_data+'Y_miml.pckl', 'wb')
pickle.dump(Y_miml, f)
f.close()
print("MIML data saved.")