# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 21:59:16 2017

@author: josemiguelarrieta
"""

import numpy as np

def evaluationEnsemble(truelab,outlab):
    """
    Example: 
        truelab = np.array([1,0,1,0,0])  
        truelab = truelab.reshape(len(truelab),1) 
        outlab = np.array([1,0,1,0,0])
        outlab = outlab.reshape(len(outlab),1)
        metrics = evaluationEnsemble(truelab,outlab)
    """
    
    It = np.where(truelab == 1)[0]
    Io = np.where(truelab == 0)[0]
    
    TP = np.sum(outlab[It])
    FN = np.sum(np.logical_not(outlab[It])) 
    
    FP  = np.sum(outlab[Io])
    TN  = np.sum(np.logical_not(outlab[Io])) 
    
    
    P = float(TP)/(TP+FP)
    R = float(TP)/(TP+FN)
    F = 2*(R*P)/(R+P)
    G = np.sqrt(float(R)*(float(TN)/(TN+FP)))
    
    tpr = float(TP)/(TP+FN)
    fpr = float(FP)/(FP+TN)
    e = (float(FP)+FN)/(FP+FN+TP+TN)
    
    if F is None:
        F = 0
        
    if G is None:
        G = 0
        
    if P is None:
        P = 0
        
    if R is None:
        R = 0
        
    AUC = (tpr*fpr)/2 + tpr*(1-fpr) + ((1-tpr)*(1-fpr))/2
    MCD = ((1-e) + 2*AUC + 2*F + 2*G)/7 
    
    metrics = [e, TP, FN, FP, TN, P, R, F, G, AUC, MCD]
    
    return metrics