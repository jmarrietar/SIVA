# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:00:49 2017

@author: josemiguelarrieta
"""
from sklearn.datasets import make_multilabel_classification
from sklearn.semi_supervised import label_propagation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_circles

#SISL - Semi-supervised {Graph}

n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False)
outer, inner = 0, 1
labels = -np.ones(n_samples)
labels[0] = outer
labels[-1] = inner

label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=1.0)
label_spread.fit(X, labels)

output_labels = label_spread.transduction_
ans = label_spread.predict(X)

"""
Active learning =
    ->http://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits_active_learning.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-digits-active-learning-py
"""

# SIML - Multilabel {oneVSRest}
X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=False,
                                      random_state=1)                                      
random_state = np.random.RandomState(0)                                
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=random_state)
classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X_train, y_train)
prediction = classif.predict(X_test)    
prediction2 = prediction[:,[0]] * prediction[:,[1]]         

# MISL
import sys,os
from sklearn import cross_validation
import random as rand
from sklearn.utils import shuffle
os.chdir('/Users/josemiguelarrieta/Documents/MILpy')
sys.path.append(os.path.realpath('..'))
from data import load_data
from MILpy.Algorithms.simpleMIL import simpleMIL
bags,labels,_ = load_data('data_gauss')  #Gaussian data
seed = 66


#Shuffle Data
bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))
train_bags, test_bags, train_labels, test_labels = cross_validation.train_test_split(bags, labels, test_size=0.1, random_state=seed)

SMILa = simpleMIL() 
SMILa.fit(train_bags, train_labels, type='average')
predictions = SMILa.predict(test_bags)



#MIML[MATLAB]

