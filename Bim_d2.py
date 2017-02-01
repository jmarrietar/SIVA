# -*- coding: utf-8 -*-

from scipy import signal
import numpy as np

def Bim_d2(X):
    """
    J = Bim_d2(I)
    
    Toolbox: Balu
    Second derivative of image X.
    
    Input data:
        I grayvalue image.
    
    Output:
        J = signal.convolve2d(I,np.array([[0,1,0],[1,-4,1],[0,1,0]]),'same');
    
    Example:
        import numpy as np
        from balu.ImagesAndData import balu_imageload
        from mahotas.colors import rgb2gray
        X = balu_imageload('testimg2.jpg')
        I = rgb2gray(X,dtype=np.uint8)
        J = Bim_d2(I);
        plt.imshow(J,cmap='gray')


        (c) D.Mery, PUC-DCC, 2010
        http://dmery.ing.puc.cl
        
        With collaboration from:
        Jose Miguel Arrieta Ramos (jmarrietar@unal.edu.co) -> Translated implementation into python (2017)
        
    """
    
    return signal.convolve2d(X,np.array([[0,1,0],[1,-4,1],[0,1,0]]),'same')

