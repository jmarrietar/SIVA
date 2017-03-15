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
    
    try:
        P = float(TP)/(TP+FP)
    except ZeroDivisionError:
        print "Oops!  That was no valid number."
        P = 0
    try:  
        R = float(TP)/(TP+FN)
    except ZeroDivisionError:
        print "Oops!  That was no valid number."
        R = 0 
    try: 
        F = 2*(R*P)/(R+P)
    except ZeroDivisionError:
        print "Oops!  That was no valid number."
        F = 0
    try: 
        G = np.sqrt(float(R)*(float(TN)/(TN+FP)))
    except ZeroDivisionError:
        print "Oops!  That was no valid number."
        G = 0 
    try: 
        tpr = float(TP)/(TP+FN)
    except ZeroDivisionError:
        tpr = 0 
    try: 
        fpr = float(FP)/(FP+TN)
    except ZeroDivisionError:
        print "Oops!  That was no valid number."
        fpr = 0 
    try: 
        e = (float(FP)+FN)/(FP+FN+TP+TN)
    except ZeroDivisionError:
        print "Oops!  That was no valid number."
        e = 0 
    
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