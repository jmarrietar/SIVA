# -*- coding: utf-8 -*-
"""
Cross Validation for differents Algorithms paradigms. 

@author: josemiguelarrieta
"""
import os
os.chdir('Documents/SIVA')
from sklearn.cross_validation import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from utils_dagm import RemodeNanInstances
from utils_dagm import LoadDataEvaluation
from sklearn.semi_supervised import label_propagation
from Evaluation import evaluationEnsemble
import numpy as np
import sys

#General Information 
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'
ClassNumber = 1
number_experimet = 1
folds = 10
                                        #------#
                                        #-SISL-#
                                        #------#
InstanceType = 'SISL'
LabelType = 'ALL'
defect = 'AB'

X_sisl,Y_sisl = LoadDataEvaluation(LabelType,defect,ClassNumber,InstanceType)

#remove rows with nan columns 
nanrows = np.unique(np.where(np.isnan(X_sisl))[0])
X_sisl = np.delete(X_sisl, nanrows, 0)
Y_sisl = np.delete(Y_sisl, nanrows, 0)

#Sin etiquetas
rng = np.random.RandomState(0)
Y_sisl[rng.rand(len(Y_sisl)) < 0.3] = -1

label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=1.0)

folds = 10
skf = StratifiedKFold(Y_sisl.reshape(len(Y_sisl)), n_folds=folds)
results = [] 
AUC = []
F = []
fold = 1

for train_index, test_index in skf:
    X_train = X_sisl[train_index]
    Y_train = Y_sisl[train_index]
    X_test  = X_sisl[test_index]
    Y_test  = Y_sisl[test_index]
    sys.stdout.write('Fold# '+str(fold)+'...')
    label_spread.fit(X_train, Y_train)
    predictions = label_spread.predict(X_test) 
    try:
        metrics = evaluationEnsemble(truelab=Y_test,outlab=predictions)
    except ZeroDivisionError:
        print "Oops!  That was no valid number.  Try again..."
        metrics =[0,0,0,0,0,0,0,0,0,0,0]
    AUC.append(metrics[9])
    F.append(metrics[7])
    results.append(metrics)
    fold +=1
    
target = open('Results.txt', 'a')
target.write('Results '+InstanceType+"\n"+'Class'+str(ClassNumber)+"\n"+'F= '+str(np.mean(F))+"\n"+ 'AUC ='+str(np.mean(AUC))+"\n")
target.close()

                                        #------#
                                        #-SIML-#
                                        #------#
#Multilabel {oneVSRest}
LabelType = 'ALL'
InstanceType = 'SIML'
defect = 'AB'

X_siml,Y_siml = LoadDataEvaluation(LabelType,defect,ClassNumber,InstanceType)


#remove rows with nan columns 
nanrows = np.unique(np.where(np.isnan(X_siml))[0])
X_siml = np.delete(X_siml, nanrows, 0)
Y_siml = np.delete(Y_siml, nanrows, 0)


labelsAB = Y_siml[:,[0]] * Y_siml[:,[1]]
skf = StratifiedKFold(labelsAB.reshape(len(labelsAB)), n_folds=folds)
folds = 1
results = [] 
AUC = []   
F = []
classif = OneVsRestClassifier(SVC(kernel='linear'))

for train_index, test_index in skf:
    X_train = X_siml[train_index]
    Y_train = Y_siml[train_index]
    X_test  = X_siml[test_index]
    Y_test  = Y_siml[test_index]
    sys.stdout.write('Fold# '+str(fold)+'...')
    classif.fit(X_train, Y_train)
    predictions = classif.predict(X_test)
    try:
        predictions2sl = Y_test[:,[0]] * Y_test[:,[1]]
        Ytest2sl = predictions[:,[0]] * predictions[:,[1]]
        metrics = evaluationEnsemble(truelab=Ytest2sl,outlab=predictions2sl)
    except ZeroDivisionError:
        print "Oops!  That was no valid number.  Try again..."
        metrics =[0,0,0,0,0,0,0,0,0,0,0]
    AUC.append(metrics[9])
    F.append(metrics[7])
    results.append(metrics)
    fold +=1
    
target = open('Results.txt', 'a')
target.write('Results '+InstanceType+"\n"+'Class'+str(ClassNumber)+"\n"+'F= '+str(np.mean(F))+"\n"+ 'AUC ='+str(np.mean(AUC))+"\n")
target.close()

    
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

X_misl,Y_misl = LoadDataEvaluation(LabelType,defect,ClassNumber,InstanceType)


#remove rows with nan columns
X_misl = RemodeNanInstances(X_misl)

folds = 10 

skf = StratifiedKFold(Y_misl.reshape(len(Y_misl)), n_folds=folds)
fold = 1
results = [] 
AUC = []
F = []
SMILa = simpleMIL() 

for train_index, test_index in skf:
    X_train = [X_misl[i] for i in train_index]
    Y_train = Y_misl[train_index]
    X_test  = [X_misl[i] for i in test_index]
    Y_test  = Y_misl[test_index]
    sys.stdout.write('Fold# '+str(fold)+'...')
    SMILa.fit(X_train, Y_train, type='average')
    predictions = SMILa.predict(X_test)
    try:
        metrics = evaluationEnsemble(truelab=Y_test,outlab=predictions)
    except ZeroDivisionError:
        print "Oops!  That was no valid number.  Try again..."
        metrics =[0,0,0,0,0,0,0,0,0,0,0]
    AUC.append(metrics[9])
    F.append(metrics[7])
    results.append(metrics)
    fold +=1

target = open('Results.txt', 'a')
target.write('Results '+InstanceType+"\n"+'Class'+str(ClassNumber)+"\n"+'F= '+str(np.mean(F))+"\n"+ 'AUC ='+str(np.mean(AUC))+"\n")
target.close()

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

X_miml,Y_miml = LoadDataEvaluation(LabelType,defect,ClassNumber,InstanceType)

#remove rows with nan columns 
X_miml = RemodeNanInstances(X_miml)

folds = 10 
labelsAB = Y_miml[:,[0]] * Y_miml[:,[1]]
skf = StratifiedKFold(labelsAB.reshape(len(labelsAB)), n_folds=folds)
prueba = []
test = []
fold = 1

for train_index, test_index in skf:
    X_train = [X_miml[i] for i in train_index]
    Y_train = Y_miml[train_index]
    X_test  = [X_miml[i] for i in test_index]
    Y_test  = Y_miml[test_index]
    sys.stdout.write('Fold# '+str(fold)+'...')
    sio.savemat('ExperimentsData/folds_miml/MIML'+'_'+str(ClassNumber)+'fold_'+str(fold)+'.mat', {'X_train':X_train,'Y_train':Y_train,'X_test':X_test,'Y_test':Y_test})
    target = open('Results.txt', 'a')
    target.write('MIML'+'_'+str(ClassNumber)+'fold_'+str(fold)+'.mat'+' CREATED'+"\n")
    target.close()
    fold +=1