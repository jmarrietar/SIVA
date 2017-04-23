#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 10:14:59 2017

@author: josemiguelarrieta
"""

from __future__ import print_function
import os
os.chdir('Documents/SIVA')
import pickle

ClassNumber = 1  #Pass argument!. 

#Load Data SIngle Label 
LabelType = 'SL'  
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+LabelType+'/'
f = open(path_data+'BagsSL.pckl', 'rb') 
BagsSL = pickle.load(f)
f.close()
f = open(path_data+'BagLabelsSL.pckl', 'rb') 
BagLabelsSL = pickle.load(f)
f.close()
f = open(path_data+'InstanceBagLabelSL.pckl', 'rb') 
InstanceBagLabelSL = pickle.load(f)
f.close()

#Load Data Multi Label
LabelType = 'ML'
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+LabelType+'/'
f = open(path_data+'BagsML.pckl', 'rb') 
BagsML = pickle.load(f)
f.close()
f = open(path_data+'BagLabelsML.pckl', 'rb') 
BagLabelsML = pickle.load(f)
f.close()
f = open(path_data+'InstanceBagLabelML.pckl', 'rb') 
InstanceBagLabelML = pickle.load(f)
f.close()