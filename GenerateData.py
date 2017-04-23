#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Generate Data for Experimentation.
    -> Data is created in a way to unify  Single and Multi instance. 
    -> SISL and MISL share the same data format.
    -> SIML and MIML share the same data format.

@author: josemiguelarrieta
"""

from __future__ import print_function
import sys
import numpy as np
import pickle
from utils_dagm import WeakLabeling,load_image_dagm
from LabelExtractFeatures import get_data_MISL, get_data_MIML

#Path
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'
defects =['AB','NO']

ClassNumber = sys.argv[1]

        ####################
        #   Single Label   #
        ####################

LabelType = 'SL'       
BagsSL = []
BagLabelsSL = np.empty((0,1), int)
InstanceBagLabelSL =[]

for defect in defects:
    for i in range (1,100):
        num = i    #Image number 
        if defect == 'AB':
            cropped,cropped_maskA,cropped_maskB = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            baglabel, bag, instanceLabels = get_data_MISL(cropped,cropped_maskA,cropped_maskB)
        elif defect == 'NO':
            image = load_image_dagm(path,num,ClassNumber,defect = defect,exp = True)
            baglabel, bag, instanceLabels = get_data_MISL(image)
        else:
            cropped,cropped_mask = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            baglabel, bag, instanceLabels = get_data_MISL(cropped,cropped_mask)
        BagsSL.append(bag)
        InstanceBagLabelSL.append(instanceLabels)
        BagLabelsSL = np.concatenate((BagLabelsSL, np.array([baglabel]).reshape(1,1)), axis=0)

#Save Data
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+LabelType+'/'
f = open(path_data+'BagsSL.pckl', 'wb')
pickle.dump(BagsSL, f)
f.close()
f = open(path_data+'BagLabelsSL.pckl', 'wb')
pickle.dump(BagLabelsSL, f)
f.close()
f = open(path_data+'InstanceBagLabelSL.pckl', 'wb')
pickle.dump(InstanceBagLabelSL, f)
f.close()
print("SL data saved.")

        ###################
        #   Multi Label   #
        ###################

LabelType = 'ML'
BagsML = []
BagLabelsML = np.empty((0,2), int)
InstanceBagLabelML = []

for defect in defects:
    for i in range (1,100):
        num = i     #Image number 
        if defect == 'AB':    
            cropped,cropped_maskA,cropped_maskB = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            baglabels, bag, instancelabels = get_data_MIML(cropped,cropped_maskA,cropped_maskB)
        elif defect == 'NO':
            image = load_image_dagm(path,num,ClassNumber,defect = defect,exp = True)
            baglabels, bag, instancelabels = get_data_MIML(image,cropped_maskA=None,cropped_maskB=None)  
        elif defect == 'A':
            cropped,cropped_maskA = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            baglabels, bag, instancelabels = get_data_MIML(cropped,cropped_maskA=cropped_maskA,cropped_maskB=None)
        elif defect == 'B':
            cropped,cropped_maskB = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
            baglabels, bag, instancelabels = get_data_MIML(cropped,cropped_maskA=None,cropped_maskB=cropped_maskB)
        BagsML.append(bag)
        InstanceBagLabelML.append(instancelabels)
        BagLabelsML = np.concatenate((BagLabelsML, baglabels), axis=0)
    
#Save Data
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+LabelType+'/'
f = open(path_data+'BagsML.pckl', 'wb')
pickle.dump(BagsML, f)
f.close()
f = open(path_data+'BagLabelsML.pckl', 'wb')
pickle.dump(BagLabelsML, f)
f.close()
f = open(path_data+'InstanceBagLabelML.pckl', 'wb')
pickle.dump(InstanceBagLabelML, f)
f.close()
print("ML data saved.")