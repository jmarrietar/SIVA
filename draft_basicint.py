# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 12:43:39 2017

@author: josemiguelarrieta
"""
import numpy as np
import matplotlib.pyplot as plt
from mahotas.colors import rgb2gray
from balu.ImageProcessing import Bim_segbalu 
from balu.ImagesAndData import balu_imageload
import os
os.chdir('Documents/SIVA')
from Bfx_basicint import Bfx_basicint
from Bim_d1 import Bim_d1
from Bim_d2 import Bim_d2

#Bfx_basic_int
I = balu_imageload('/Users/josemiguelarrieta/Documents/MATLAB/beach.jpg')
I = rgb2gray(I,dtype=np.uint8)
R,_,_ = Bim_segbalu(I)
options = {'show': True, 'mask': 5}
X, Xn = Bfx_basicint(I,R,options)


#Dim1
import numpy as np
from balu.ImagesAndData import balu_imageload
from mahotas.colors import rgb2gray
X = balu_imageload('/Users/josemiguelarrieta/Documents/MATLAB/beach.jpg')
I = rgb2gray(X,dtype=np.uint8)
J,_,_ = Bim_d1(I,5);
plt.imshow(J,cmap='gray')

#Dim2
import numpy as np
from balu.ImagesAndData import balu_imageload
from mahotas.colors import rgb2gray
X = balu_imageload('/Users/josemiguelarrieta/Documents/MATLAB/beach.jpg')
I = rgb2gray(X,dtype=np.uint8)
J = Bim_d2(I);
plt.imshow(J,cmap='gray')



