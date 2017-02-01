# -*- coding: utf-8 -*-

from scipy import signal
import numpy as np

def Bim_d1(X,m):
    """
    J,_,_ = Bim_d1(I,m);
    
    Toolbox: Balu
    First derivative of image X using a m x m Gauss operator.
    
    Input data:
        I grayvalue image.
    
    Output:
        J first derivative of I
        
    Example:
        import numpy as np
        from balu.ImagesAndData import balu_imageload
        from mahotas.colors import rgb2gray
        X = balu_imageload('testimg2.jpg')
        I = rgb2gray(X,dtype=np.uint8)
        J,_,_ = Bim_d1(I,5);
        plt.imshow(J,cmap='gray')
        
    (c) D.Mery, PUC-DCC, 2010
    http://dmery.ing.puc.cl
        
    With collaboration from:
    Jose Miguel Arrieta Ramos (jmarrietar@unal.edu.co) -> Translated implementation into python (2017)
    """
    
    sigma = m/8.5
    s2 = sigma**2
    Gx = np.zeros((m,m))
    Gy = np.zeros((m,m))
    c = (m-1)/2
    for i in range (0,m):
       x = i+1-c
       x2 = (i+1-c)**2
       for j in range (0,m):
          y = j+1-c
          y2 = (j+1-c)**2
          ex = np.exp(-(x2+y2)/2.0/s2)   
          Gx[i,j] = y*ex
          Gy[i,j] = x*ex
    mgx = np.sum(np.abs(np.asarray(Gx).ravel()))/2.0*(0.3192*m-0.3543)
    Gx = Gx/mgx
    Gy = Gy/mgx
    Yx = signal.convolve2d(X,Gx,'same');
    Yy = signal.convolve2d(X,Gy,'same');
    Y0 = np.sqrt(Yx*Yx+Yy*Yy)
    N,M = X.shape
    Y = np.zeros((N,M))
    Y[c-1:N-c-1,c-1:M-c-1] = Y0[c-1:N-c-1,c-1:M-c-1]
    return Y,Yx,Yy


