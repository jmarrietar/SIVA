# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 20:26:49 2017

@author: josemiguelarrieta
"""
from __future__ import print_function
import os
os.chdir('Documents/SIVA')
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import label_propagation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from utils_dagm import RemodeNanInstances

#General Information 
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'
ClassNumber = 1
number_experimet = 1
                                        #------#
                                        #-SISL-#
                                        #------#
InstanceType = 'SISL'
LabelType = 'ALL'
defect = 'AB'

#Load Data
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData/'+LabelType+'/defect'+defect+'/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'X_sisl.pckl', 'rb') 
X_sisl = pickle.load(f)
f.close()
f = open(path_data+'Y_sisl.pckl', 'rb') 
Y_sisl = pickle.load(f)
f.close()

#remove rows with nan columns 
nanrows = np.unique(np.where(np.isnan(X_sisl))[0])
X_sisl = np.delete(X_sisl, nanrows, 0)
Y_sisl = np.delete(Y_sisl, nanrows, 0)

#Label Spreading. (Entrena con datos unlabeled!)
X_train, X_test, y_train, y_test = train_test_split(X_sisl, Y_sisl, test_size=0.33, random_state=42)

#Sin etiquetas
rng = np.random.RandomState(0)
y_train[rng.rand(len(y_train)) < 0.3] = -1

label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=1.0)
label_spread.fit(X_train, y_train)

prediction = label_spread.predict(X_test) 

                                        #------#
                                        #-SIML-#
                                        #------#
#Multilabel {oneVSRest}
LabelType = 'ALL'
InstanceType = 'SIML'
defect = 'AB'
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData/'+LabelType+'/defect'+defect+'/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'

#Load Data
f = open(path_data+'X_siml.pckl', 'rb') 
X_siml = pickle.load(f)
f.close()
f = open(path_data+'Y_siml.pckl', 'rb') 
Y_siml = pickle.load(f)
f.close()

#remove rows with nan columns 
nanrows = np.unique(np.where(np.isnan(X_siml))[0])
X_siml = np.delete(X_siml, nanrows, 0)
Y_siml = np.delete(Y_siml, nanrows, 0)

#Split Data
X_train, X_test, y_train, y_test = train_test_split(X_siml, Y_siml, test_size=0.33, random_state=42)

classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X_train, y_train)
prediction = classif.predict(X_test)    
prediction2 = prediction[:,[0]] * prediction[:,[1]]


                                        #------#
                                        #-MISL-#
                                        #------#
import sys,os
os.chdir('/Users/josemiguelarrieta/Documents/MILpy')
sys.path.append(os.path.realpath('..'))
from MILpy.Algorithms.simpleMIL import simpleMIL
seed = 66

LabelType = 'ALL'
InstanceType = 'MISL'
defect = 'AB'
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData/'+LabelType+'/defect'+defect+'/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'

#Load Data
f = open(path_data+'Bags_misl.pckl', 'rb') 
X_misl = pickle.load(f)
f.close()
f = open(path_data+'Y_misl.pckl', 'rb') 
Y_misl = pickle.load(f)

#remove rows with nan columns
X_misl = RemodeNanInstances(X_misl)

#Split Data
X_train, X_test, y_train, y_test = train_test_split(X_misl, Y_misl, test_size=0.33, random_state=42)


SMILa = simpleMIL() 
SMILa.fit(X_train, y_train, type='average')
predictions = SMILa.predict(X_test)

                                        #------#
                                        #-MIML-#
                                        #------#

"""
MIML is done in MATLAB. 
Here Data Transformation is done to export to Matlab format. 
"""
LabelType = 'ALL'
InstanceType = 'MIML'
defect = 'AB'
import scipy.io as sio


path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData/'+LabelType+'/defect'+defect+'/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'

#Load Data
f = open(path_data+'Bags_miml.pckl', 'rb') 
X_miml = pickle.load(f)
f.close()
f = open(path_data+'Y_miml.pckl', 'rb') 
Y_miml = pickle.load(f)
f.close()

sio.savemat('X_miml.mat', {'X_miml':X_miml})
sio.savemat('Y_miml.mat', {'Y_miml':Y_miml})

sio.savemat('MIML.mat', {'X_miml':X_miml,'Y_miml':Y_miml})


