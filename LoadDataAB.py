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
defects =['AB','NO']

#Load Data
InstanceType = 'SISL'
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'X_sisl.pckl', 'rb') 
X_sisl = pickle.load(f)
f.close()
f = open(path_data+'Y_sisl.pckl', 'rb') 
Y_sisl = pickle.load(f)
f.close()

#Load Data
InstanceType = 'MISL'
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'Bags_misl.pckl', 'rb') 
X_misl = pickle.load(f)
f.close()
f = open(path_data+'Y_misl.pckl', 'rb') 
Y_misl = pickle.load(f)
f.close()

#Load Data
InstanceType = 'SIML'
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'X_siml.pckl', 'rb') 
X_siml = pickle.load(f)
f.close()
f = open(path_data+'Y_siml.pckl', 'rb') 
Y_siml = pickle.load(f)
f.close()

#Load Data
InstanceType = 'MIML'
path_data = '/Users/josemiguelarrieta/Documents/SIVA/ExperimentsData2/defectAB/'+'class'+str(ClassNumber)+'/'+InstanceType+'/'
f = open(path_data+'Bags_miml.pckl', 'rb') 
Bags_miml = pickle.load(f)
f.close()
f = open(path_data+'Y_miml.pckl', 'rb') 
Y_miml = pickle.load(f)
f.close()
