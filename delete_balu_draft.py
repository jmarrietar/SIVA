# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:17:15 2017

@author: josemiguelarrieta
"""
from sklearn import datasets

from balu.ImagesAndData import balu_load
from balu.Classification import Bcl_lda
from balu.InputOutput import Bio_plotfeatures
from balu.PerformanceEvaluation import Bev_performance

data = balu_load('datagauss')           #simulated data (2 classes, 2 features)
X = data['X']
d = data['d']
Xt = data['Xt']
dt = data['dt']
Bio_plotfeatures(X, d)                  # plot feature space
op = {'p': []}
ds, options = Bcl_lda(X, d, Xt, op)     # LDA classifier
p = Bev_performance(ds, dt)             # performance on test data
print p


##################
#### Mahotas #####
##################

import mahotas as mh
import mahotas.features
import mahotas.center_of_mass as center_of_mass
image3 = mh.imread('Cl_2_2_AB.png')
#mh.imsave('copy.png', image3)

ans = mahotas.features.haralick(image3).mean(0)
ans2 = mahotas.features.pftas(image3)
mahotas.features.zernike(image3, radius=2,cm={center_of_mass(image3)})