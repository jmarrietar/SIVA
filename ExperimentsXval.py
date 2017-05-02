# -*- coding: utf-8 -*-
"""
Cross Validation for differents Algorithms paradigms. 

@author: josemiguelarrieta

"""
from sklearn.cross_validation import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from utils_dagm import RemodeNanInstances
from utils_dagm import LoadDataEvaluation
from sklearn.semi_supervised import label_propagation
from Evaluation import evaluationEnsemble
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
import numpy as np
import sys

#General Information 
path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'
ClassNumber = 5
folds = 10
                                        ###########
                                        #   SISL  #
                                        ###########
                                    
                                        
#Load data
LabelType = 'SL'
paradigm = 'SISL'
X_sisl,Y_sisl,instance_label_bags_SL = LoadDataEvaluation(ClassNumber,LabelType)

#Suffle
X_sisl, instance_label_bags_SL, Y_sisl = shuffle(X_sisl, instance_label_bags_SL, Y_sisl, random_state=0)

#remove rows with nan columns
X_sisl,instance_label_bags_SL = RemodeNanInstances(X_sisl,instance_label_bags_SL) 

#Normalize Bags
X_sisl = [normalize(bag,norm = 'l2',axis=0) for bag in X_sisl]

kernel='knn'
#kernel = 'rbf'
label_spread = label_propagation.LabelSpreading(kernel=kernel, alpha=1.0)


skf = StratifiedKFold(Y_sisl.reshape(len(Y_sisl)), n_folds=folds, shuffle=True)

results = [] 
AUC = []
F = []
fold = 1
Predictions = np.empty((0,1), int)
    
for train_index, test_index in skf:
    X_train = [X_sisl[i] for i in train_index]
    Y_train = Y_sisl[train_index]
    instance_label_bags_SL_Train = [instance_label_bags_SL[i] for i in train_index]    
    X_test  = [X_sisl[i] for i in test_index]
    Y_test  = Y_sisl[test_index]
    instance_label_bags_SL_Test = [instance_label_bags_SL[i] for i in test_index] 
    
    sys.stdout.write('Fold# '+str(fold)+'...')
    #Bags to SI
    X_train_all = np.vstack((X_train))
    instance_label_bags_SL_Train_ALL = np.vstack((instance_label_bags_SL_Train))
    
    #Remove labels 30%
    rng = np.random.RandomState(0)
    instance_label_bags_SL_Train_ALL[rng.rand(len(instance_label_bags_SL_Train_ALL)) < 0.3] = -1
    
    #Fit model. 
    label_spread.fit(X_train_all, instance_label_bags_SL_Train_ALL)
    
    #Test 
    for i in range(len(X_test)):
        predictions_instances = label_spread.predict(X_test[i]) 
        if 1 in predictions_instances:
            Predictions= np.append(Predictions,np.array([[1]]), axis = 0)
        else : 
            Predictions= np.append(Predictions,np.array([[0]]), axis = 0)
    #Metrics
    metrics = evaluationEnsemble(truelab=Y_test,outlab=Predictions)
    AUC.append(metrics[9])
    F.append(metrics[7])
    results.append(metrics)
    fold +=1
    
target = open('Results.txt', 'a')
target.write('Results '+paradigm+' kernel:'+kernel+"\n"+'Class'+str(ClassNumber)+"\n"+'F= '+str(np.mean(F))+"\n"+ 'AUC ='+str(np.mean(AUC))+"\n")
target.close()

                                        ###########
                                        #   SIML  #
                                        ###########
#Multilabel {oneVSRest}
LabelType = 'ML'
paradigm = 'SIML'

#Load Data
X_siml,Y_siml,instance_label_bags_ML = LoadDataEvaluation(ClassNumber,LabelType)

#Suffle
X_siml, instance_label_bags_ML, Y_siml = shuffle(X_siml, instance_label_bags_ML, Y_siml, random_state=0)

#remove rows with nan columns 
X_siml,instance_label_bags_ML = RemodeNanInstances(X_siml,instance_label_bags_ML) 

#Normalize 
X_siml = [normalize(bag,norm = 'l2',axis=0) for bag in X_siml]

labelsAB = np.logical_or(Y_siml[:,[0]],Y_siml[:,[1]])
labelsAB = labelsAB.astype(int)

skf = StratifiedKFold(labelsAB.reshape(len(labelsAB)), n_folds=folds, shuffle=True)

results = [] 
AUC = []   
F = []
fold = 1
#kernel='linear'
kernel = 'rbf'
classif = OneVsRestClassifier(SVC(kernel=kernel))
Predictions = np.empty((0,1), int)

for train_index, test_index in skf:
    X_train = [X_siml[i] for i in train_index]
    Y_train = Y_siml[train_index]
    instance_label_bags_ML_Train = [instance_label_bags_ML[i] for i in train_index]    
    X_test  = [X_siml[i] for i in test_index]
    Y_test  = Y_siml[test_index]
    instance_label_bags_ML_Test = [instance_label_bags_ML[i] for i in test_index] 
    sys.stdout.write('Fold# '+str(fold)+'...')
    
    #Bags to SI
    X_train_all = np.vstack((X_train))
    instance_label_bags_ML_Train_ALL = np.vstack((instance_label_bags_ML_Train))
    
    #Fit Model
    classif.fit(X_train_all, instance_label_bags_ML_Train_ALL)
    
    #Test 
    for i in range(len(X_test)):
        predictions_instances = classif.predict(X_test[i]) 
        predictions_instances2SL = np.logical_or(predictions_instances[:,[0]],predictions_instances[:,[1]])
        predictions_instances2SL = predictions_instances2SL.astype(int)
        if 1 in predictions_instances2SL:
            Predictions= np.append(Predictions,np.array([[1]]), axis = 0)
        else : 
            Predictions= np.append(Predictions,np.array([[0]]), axis = 0)
    Ytest2sl = np.logical_or(Y_test[:,[0]],Y_test[:,[1]])
    Ytest2sl = Ytest2sl.astype(int)
    metrics = evaluationEnsemble(truelab=Ytest2sl,outlab=Predictions)
    AUC.append(metrics[9])
    F.append(metrics[7])
    results.append(metrics)
    fold +=1
    
target = open('Results.txt', 'a')
target.write('Results '+paradigm+' kernel:'+kernel+"\n"+'Class'+str(ClassNumber)+"\n"+'F= '+str(np.mean(F))+"\n"+ 'AUC ='+str(np.mean(AUC))+"\n")
target.close()

    

'''
Estos Multi Instance puede que se hallan Dañado por el 
valor adicional que la funcion tiene [Arreglarlo].
'''
                                        ############
                                        #   MISL   #
                                        ############
import sys,os
sys.path.append(os.path.realpath('..'))
from MILpy.Algorithms.simpleMIL import simpleMIL
from MILpy.Algorithms.CKNN import CKNN
from MILpy.Algorithms.EMDD import EMDD
from MILpy.Algorithms.BOW import BOW
seed = 66

LabelType = 'SL'
paradigm = 'MISL'

#Load Data
X_misl,Y_misl,instance_label_bags_ML = LoadDataEvaluation(ClassNumber,LabelType)

#Suffle
X_misl, instance_label_bags_ML, Y_misl = shuffle(X_misl, instance_label_bags_ML, Y_misl, random_state=0)

#remove rows with nan columns
X_misl,instance_label_bags_ML = RemodeNanInstances(X_misl,instance_label_bags_ML)

#Normalize Bags
X_misl = [normalize(bag,norm = 'l2',axis=0) for bag in X_misl]

skf = StratifiedKFold(Y_misl.reshape(len(Y_misl)), n_folds=folds, shuffle=True)
fold = 1
results = [] 
AUC = []
F = []

#Algorithms
SMILa = simpleMIL()
#SMILa = BOW()
#SMILa = EMDD() 

'''
NOTE: SimpleMIL devuelve 1 valor
los demas devuelven 2 
(Modificar Acorde).
'''
for train_index, test_index in skf:
    X_train = [X_misl[i] for i in train_index]
    Y_train = Y_misl[train_index]
    X_test  = [X_misl[i] for i in test_index]
    Y_test  = Y_misl[test_index]
    sys.stdout.write('Fold# '+str(fold)+'...')
    SMILa.fit(X_train, Y_train, type='min') #SimpleMIL {'average','extreme','max', 'min'}
    #SMILa.fit(X_train, Y_train) #EMDD 
    #SMILa.fit(X_train, Y_train,references = 3, citers = 5) #CNN
    #SMILa.fit(X_train, Y_train, k=10, covar_type = 'diag')#BOW
    predictions = SMILa.predict(X_test)
    if type(predictions) is tuple:
        metrics = evaluationEnsemble(truelab=Y_test,outlab=predictions[0])
    else:
        metrics = evaluationEnsemble(truelab=Y_test,outlab=predictions)
    AUC.append(metrics[9])
    F.append(metrics[7])
    results.append(metrics)
    fold +=1

os.chdir('../../Documents/SIVA')
target = open('Results.txt', 'a')
target.write('Results '+paradigm+"\n"+'Class'+str(ClassNumber)+' Algo:'+str(SMILa)+"\n"+'F= '+str(np.mean(F))+"\n"+ 'AUC ='+str(np.mean(AUC))+"\n")
target.close()


                                        ###########
                                        #   MIML  #
                                        ###########

"""
MIML is done in MATLAB. 
Here Data Transformation is done to export to Matlab format. 
"""
LabelType = 'ML'
paradigm = 'MIML'
defect = 'AB'

import scipy.io as sio

#Load data
X_miml,Y_miml,instance_label_bags_ML = LoadDataEvaluation(ClassNumber,LabelType)

#Suffle
X_miml, instance_label_bags_ML, Y_miml = shuffle(X_miml, instance_label_bags_ML, Y_miml, random_state=0)

#remove rows with nan columns 
X_miml,instance_label_bags_ML = RemodeNanInstances(X_miml,instance_label_bags_ML) 

#Normalize Bags
X_miml = [normalize(bag,norm = 'l2',axis=0) for bag in X_miml]

labelsAB = np.logical_or(Y_miml[:,[0]],Y_miml[:,[1]])
labelsAB = labelsAB.astype(int)

skf = StratifiedKFold(labelsAB.reshape(len(labelsAB)), n_folds=folds,shuffle=True)
prueba = []
test = []
fold = 1

for train_index, test_index in skf:
    X_train = [X_miml[i] for i in train_index]
    Y_train = Y_miml[train_index]
    X_test  = [X_miml[i] for i in test_index]
    Y_test  = Y_miml[test_index]
    sys.stdout.write('Fold# '+str(fold)+'...')
    sio.savemat('ExperimentsData2/folds_miml/class'+str(ClassNumber)+'/class'+str(ClassNumber)+'_'+str(fold)+'.mat', {'X_train':X_train,'Y_train':Y_train,'X_test':X_test,'Y_test':Y_test,'fold':fold,'Class_number':ClassNumber})
    target = open('Results.txt', 'a')
    target.write('MIML'+'_'+str(ClassNumber)+'fold_'+str(fold)+'.mat'+' CREATED'+"\n")
    target.close()
    fold +=1