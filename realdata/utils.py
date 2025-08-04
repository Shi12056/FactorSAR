'''
This is a python file containing all functions needed in the real data analysis.
'''

import pandas as pd
import sys,codecs
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import scipy.stats as st
import collections
import statsmodels.api as sm
from time import time
import random

# CMLE-gradient
"""
Compute the gradient of the concentrated log-likelihood for a single response variable in the SAR model.

"""

# CMLE-gradient
def gardient_initial_lj(nn, p, Wc, para, Yj):
    rho_j = para[0]
    sigma2_j = para[1]
    
    S = np.eye(nn)-rho_j*Wc
    S_inverse = np.linalg.inv(S)
    G = Wc@S_inverse
    Ep = S@Yj
    
    g1 = -np.trace(G)+np.dot((Wc@Yj).T,Ep)/sigma2_j
    g2 = -nn/(2*sigma2_j)+(Ep.T@Ep)/(2*(sigma2_j**2))
    
    return np.array([g1,g2])


# FMLE-gradient
def gardient_lj(nn, p, the0, Z_e, Yj,Wc):

    rhoj = the0[0]
    betaj = the0[1]
    tauj2 = the0[2]

    S = np.eye(nn)-rhoj*Wc
    S_inverse = np.linalg.inv(S)
    G = Wc@S_inverse
    Ep = S@Yj-Z_e*betaj
    
    g1 = -np.trace(G)+np.dot((Wc@Yj).T,Ep)/tauj2
    g2 = (Z_e.T@Ep)/tauj2
    g3 = -nn/(2*tauj2)+(Ep.T@Ep)/(2*tauj2**2)
    
    return np.array([g1,g2,g3])


# CMLE-hessian
"""
Compute the Hessian matrix of the concentrated log-likelihood for a single response variable in the SAR model.

"""
def hessian_initial_lj(nn, p, Wc, para, Yj):
    
    rho_j = para[0]
    sigma2_j = para[1]
    
    S = np.eye(nn)-rho_j*Wc
    S_inverse = np.linalg.inv(S)
    G = np.dot(Wc,S_inverse)
    WWc = np.dot(Wc.T,Wc)
    WS = np.dot(Wc.T,S)
    SS = np.dot(S.T,S)
    
    
    h11 = -np.trace(G@G)-np.dot(Yj.T@WWc,Yj)/sigma2_j
    h12 = h21 = -np.dot(np.dot(Yj.T,WS),Yj)/(sigma2_j**2)
    h22 = nn/(2*sigma2_j**2)-np.dot(Yj.T@SS,Yj)/(sigma2_j**3)
    
    return np.array([[h11,h12],[h21,h22]])


# FMLE-hessian
def hessian_lj(nn, p, the0, Z_e, Yj,Wc):
    
    rhoj = the0[0]
    betaj = the0[1]
    tauj2 = the0[2]
    
    S = np.eye(nn)-rhoj*Wc
    S_inverse = np.linalg.inv(S)
    G = np.dot(Wc,S_inverse)
    WY = Wc@Yj
    Ep = S@Yj-Z_e*betaj
    
    h11 = -np.trace(G@G) - WY.T@WY/tauj2
    h12 = h21 = -Z_e.T@WY/tauj2
    h13 = h31 = -WY.T@Ep/tauj2**2
    h22 = -Z_e.T@Z_e/tauj2
    h23 = h32 = -Z_e.T@Ep/(2*tauj2**2)
    h33 = nn/(2*tauj2**2)-Ep.T@Ep/(tauj2**3)
    
    return np.array([[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]])

# Newton - iteration CMLE
"""
Newton-Raphson optimization for concentrated likelihood in SAR model.
"""
def newton_sea_initial(nn, p, pa0,Wc, Yj, max_iter = 50, eps = 1e-4):

    pa_new = pa0
    for t in range(max_iter):
        pa_pre = pa_new
        gradient = gardient_initial_lj(nn, p,Wc, pa_pre, Yj)/nn
        hessian =  hessian_initial_lj(nn, p,Wc, pa_pre, Yj)/nn 
        diff = np.linalg.inv(hessian+0.001*np.eye(2)).dot(gradient)
        pa_new = pa_pre - diff 
        if np.linalg.norm(diff) < eps:
            break
            
    return pa_new,t+1


# Newton - iteration FMLE
"""
Newton-Raphson optimization for concentrated likelihood in SAR model.
"""
def newton_sea(nn, p, theta0, Z_e, Yj, Wc,max_iter = 50, eps = 1e-4):

    theta_new = theta0
    for t in range(max_iter):
        theta_pre = theta_new
        gradient = gardient_lj(nn, p, theta_pre, Z_e, Yj,Wc)/nn 
        #print(gradient)
        hessian =  hessian_lj(nn, p, theta_pre, Z_e, Yj,Wc)/nn 
        diff = np.linalg.inv(hessian+0.0001*np.eye(3)).dot(gradient)
        theta_new = theta_pre - diff

        if np.linalg.norm(diff) < eps:
            break
            
    return theta_new,t+1

# CMLE
"""
Full maximum likelihood estimation for all p response variables.

"""
def optimize_initial(nn, p, Wc, Yc):
    
    para = np.zeros((2,p))
    ite = np.zeros(p)
    for j in range(p):
        par = np.array([0,np.var(Yc[:,j])])
        res = newton_sea_initial(nn, p, par, Wc, Yc[:,j])
        para[:,j] = res[0]
        ite[j] = res[1]
        print('initial:',j, para[0,j])
    
    return para[0,:],para[1,:],ite


# FMLE
"""
Full maximum likelihood estimation for all p response variables.

"""
def optimize(nn, p, Z_e, Y,Wc):
    
    theta = np.zeros((3,p))
    itee = np.zeros(p)
    theta0 = np.array([0,0,0.0001])
    for j in range(p):
        res = newton_sea(nn, p, theta0, Z_e, Y[:,j],Wc)
        theta[:,j] = res[0]
        itee[j] = res[1]
        print('update:',j, theta[0,j])
    
    return theta[0,:],theta[1,:], theta[2,:], itee


# Factor estimation
"""
Factor estimation function.

"""
def Z_est_new(n, n1, p,d, Y, rho_e, We, Wb, Yb, rho_hb):
    
    Eb = Yb - (Wb@Yb)@np.diag(rho_hb) 
    Sigma = Eb.T@Eb/n1
  
    U, G, V = np.linalg.svd(Sigma)
    M = U[:,:d]@np.diag(G[:d])**0.5
    
    E = Y - (We@Y)@np.diag(rho_e)
    Z_e = E@M/p
    
    return Z_e, M