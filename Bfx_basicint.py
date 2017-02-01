# -*- coding: utf-8 -*-
from mahotas import bwperim
import numpy as np
import scipy.stats.stats as st
from Bim_d1 import Bim_d1
from Bim_d2 import Bim_d2

def Bfx_basicint(I,R,*args):
    """
    X, Xn = Bfx_basicint(I,R,options)
    
    Toolbox: Balu
        Basic intensity features
    
        X is the features vector, Xn is the list feature names (see Example to see how it works).
        
        Reference:
            Kumar, A.; Pang, G.K.H. (2002): Defect detection in textured materials
            using Gabor filters. IEEE Transactions on Industry Applications,
            38(2):425-440.
        
    Example:
        import numpy as np
        from mahotas.colors import rgb2gray
        from balu.ImageProcessing import Bim_segbalu
        from balu.ImagesAndData import balu_imageload
        
        options = {'show': True, 'mask': 5}   % Gauss mask for gradient computation and display results
        I = balu_imageload(('testimg1.jpg');             % input image
        R,_,_ = Bim_segbalu(I);                     % segmentation

        X, Xn = Bfx_basicint(I,R,options)     % basic intenisty features

    See also Bfx_haralick, Bfx_clp, Bfx_gabor, Bfx_fourier, Bfx_dct, Bfx_lbp.
        
    (c) D.Mery, PUC-DCC, 2010
    http://dmery.ing.puc.cl
        
    With collaboration from:
    Jose Miguel Arrieta Ramos (jmarrietar@unal.edu.co) -> Translated implementation into python (2017)
    """
    
    if len(args) == 0:
        options = {'show': False}
    else:
        options = args[0]

    if options['show']:
        print('--- extracting Basic intensity features...')
        
    if 'mask' not in options:
        options['mask'] = 15
    
    E = bwperim(R, n=4)
    ii = R == 1
    jj = np.where(R.ravel() == 0)[0]
    kk = E==1 
    
    I = I.astype(float)
    
    I1,_,_ = Bim_d1(I,options['mask'])
    I2 = Bim_d2(I)
    
    if len(jj)>0:
        C = np.mean(np.abs(I1[kk]))
    else:
        C = -1
    
    J = I[ii]
    G = np.mean(J)
    S = np.std(J)
    K  = st.kurtosis(J,fisher=False)
    Sk = st.skew(J)
    D = np.mean(I2[ii])

    X = np.array([[
        G,
        S,
        K,
        Sk,
        D,
        C
            ]])

    Xn = [ 'Intensity Mean',
           'Intensity StdDev',
           'Intensity Kurtosis',
           'Intensity Skewness',
           'Mean Laplacian',
           'Mean Boundary Gradient']
           
    return X, Xn