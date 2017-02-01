import matplotlib
matplotlib.use('TkAgg')
from balu.ImageProcessing import Bim_d1
import matplotlib.pyplot as plt
import numpy as np
from balu.ImagesAndData import balu_imageload
from mahotas.colors import rgb2gray
X = balu_imageload('/Users/josemiguelarrieta/Documents/MATLAB/beach.jpg')
I = rgb2gray(X,dtype=np.uint8)
J,_,_ = Bim_d1(I,5);
plt.imshow(J,cmap='gray')
plt.pause(2)