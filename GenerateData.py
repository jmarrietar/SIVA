#Image Manipulation
from __future__ import print_function
import os
os.chdir('Documents/SIVA')
import numpy as np
import pickle
from utils_dagm import WeakLabeling, get_data_SISL, get_data_SIML, get_data_MISL, get_data_MIML,load_image_dagm

#Path
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'
ClassNumber = 2
number_experimet = 1
                                        #------#
                                        #-SISL-#
                                        #------#
LabelType = 'SingleLabel'
InstanceType = 'SIL'
#LabelType = 'ALL'
#defect = 'A'
#defect = 'B'
#defect = 'AB'
defect = 'NO'

X_sisl =  np.empty((0,144), float)    #58 is the number of features [Change]
Y_sisl = np.empty((0,1), int)

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
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData/'+LabelType+'/defect'+defect+'/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'X_sisl.pckl', 'wb')
pickle.dump(X_sisl, f)
f.close()
f = open(path_data+'Y_sisl.pckl', 'wb')
pickle.dump(Y_sisl, f)
f.close()

#Load Data
f = open(path_data+'X_sisl.pckl', 'rb') 
X_sisl2 = pickle.load(f)
f.close()
f = open(path_data+'Y_sisl.pckl', 'rb') 
Y_sisl2 = pickle.load(f)
f.close()

                                        #------#
                                        #-MISL-#
                                        #------#
LabelType = 'SingleLabel'
#LabelType = 'ALL'
defect = 'A'
#defect = 'B'
#defect = 'AB'
#defect = 'NO'

Bags_misl = []
Y_misl = np.empty((0,1), int)

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
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData/'+LabelType+'/defect'+defect+'/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'Bags_misl.pckl', 'wb')
pickle.dump(X_sisl, f)
f.close()
f = open(path_data+'Y_misl.pckl', 'wb')
pickle.dump(Y_sisl, f)
f.close()

#Load Data
f = open(path_data+'Bags_misl.pckl', 'rb') 
X_sisl2 = pickle.load(f)
f.close()
f = open(path_data+'Y_misl.pckl', 'rb') 
Y_sisl2 = pickle.load(f)
f.close()

                                        #------#
                                        #-SIML-#
                                        #------#
LabelType = 'MultiLabel'
#LabelType = 'ALL'
defect = 'A'
#defect = 'B'
#defect = 'AB'
#defect = 'NO'

X_siml =  np.empty((0,144), float)    #58 is the number of features [Change]
Y_siml = np.empty((0,2), int)

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
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData/'+LabelType+'/defect'+defect+'/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'X_siml.pckl', 'wb')
pickle.dump(X_sisl, f)
f.close()
f = open(path_data+'Y_siml.pckl', 'wb')
pickle.dump(Y_sisl, f)
f.close()

#Load Data
f = open(path_data+'X_siml.pckl', 'rb') 
X_sisl2 = pickle.load(f)
f.close()
f = open(path_data+'Y_siml.pckl', 'rb') 
Y_sisl2 = pickle.load(f)
f.close()

                                        #------#
                                        #-MIML-#
                                        #------#
LabelType = 'MultiLabel'
#LabelType = 'ALL'
defect = 'A'
#3defect = 'B'
#defect = 'AB'
#defect = 'NO'

Bags_miml = []
Y_miml = np.empty((0,2), int)
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
        labels_bag, bagA = get_data_MIML(cropped,cropped_maskA=cropped_maskA,cropped_maskB=None)
    elif defect == 'B':
        cropped,cropped_maskB = WeakLabeling(path,num,ClassNumber,defect = defect,exp = True)
        labels_bag, bagB = get_data_MIML(cropped,cropped_maskA=None,cropped_maskB=cropped_maskB)
    Bags_miml.append(bag)
    Y_miml = np.concatenate((Y_miml, labels_bag), axis=0)
    
#Save Data
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData/'+LabelType+'/defect'+defect+'/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'Bags_miml.pckl', 'wb')
pickle.dump(X_sisl, f)
f.close()
f = open(path_data+'Y_miml.pckl', 'wb')
pickle.dump(Y_sisl, f)
f.close()

#Load Data
f = open(path_data+'Bags_miml.pckl', 'rb') 
X_sisl2 = pickle.load(f)
f.close()
f = open(path_data+'Y_miml.pckl', 'rb') 
Y_sisl2 = pickle.load(f)
f.close()
