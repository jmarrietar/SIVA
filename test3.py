#Bfx_basic_int
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from mahotas.colors import rgb2gray
from balu.ImageProcessing import Bim_segbalu
from balu.ImagesAndData import balu_imageload
from balu.FeatureExtraction import Bfx_basicint
options = {'show': True, 'mask': 5}   
I = balu_imageload('/Users/josemiguelarrieta/Documents/MATLAB/beach.jpg') 
I = rgb2gray(I,dtype=np.uint8)           
R,_,_ = Bim_segbalu(I)                     
X, Xn = Bfx_basicint(I,R,options)     
print(X)
print(Xn)