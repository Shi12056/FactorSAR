'''
This is a python file containing all functions needed in the model selection part of simulation studies.
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



# Data generator
"""
Generate response matrix Y along with covariates X and Z based on a spatial autoregressive model.

- Y: (nn x p) response matrix
- X: (nn x q0) covariate matrix for fixed effects
- Z: (nn x d) covariate matrix for random effects
"""
def data_generator(nn, p, q, d, W, mean, cov, rho, beta, bc, seed):

    rng = np.random.default_rng(seed) 
    
    Omega = rng.multivariate_normal(mean, cov, (nn,), 'raise')   
    X = rng.normal(0,1,(nn,q))
    Z = rng.normal(0,1,(nn,d))
    
    Y = np.zeros((nn,p))
    for j in range(p):
        # Compute spatially lagged response via inverse of (I - rho * W)
        y_j = np.dot(np.linalg.inv(np.eye(nn)-rho[j]*W), X@beta[j,:] + Z@bc[j,:]+Omega[:,j])
        Y[:,j] = y_j 
        
    return Y, X, Z

# CMLE-gradient
"""
Compute the gradient of the concentrated log-likelihood for a single response variable in the SAR model.

Output:
- Gradient vector of length (q0 + 2)
"""
def gardient_initial_lj_0(nn, p, q, WW, para, Yj, XX):
    
    q0 = 10
    
    rho_j = para[0]
    beta_j = para[1:(q0+1)]
    sigma2_j = para[-1]
    
    
    Sj = np.eye(nn)-rho_j*WW
    S_inverse = np.linalg.inv(Sj)
    G = WW@S_inverse
    Ep = Sj@Yj - XX@beta_j
    
    g1 = -np.trace(G)+np.dot((WW@Yj).T,Ep)/sigma2_j
    g2 = (XX.T@Ep)/sigma2_j
    g3 = -nn/(2*sigma2_j)+(Ep.T@Ep)/(2*(sigma2_j**2))
    #print(Ep)
    
    return np.concatenate(([g1],g2,[g3]))

# CMLE-gradient-full q
"""
Compute the gradient of the concentrated log-likelihood for a single response variable in the SAR model.

Output:
- Gradient vector of length (q + 2)
"""
def gardient_initial_lj(nn, p, q, WW, para, Yj, XX):

    
    rho_j = para[0]
    beta_j = para[1:(q+1)]
    sigma2_j = para[-1]
    
    
    Sj = np.eye(nn)-rho_j*WW
    S_inverse = np.linalg.inv(Sj)
    G = WW@S_inverse
    Ep = Sj@Yj - XX@beta_j
    
    g1 = -np.trace(G)+np.dot((WW@Yj).T,Ep)/sigma2_j
    g2 = (XX.T@Ep)/sigma2_j
    g3 = -nn/(2*sigma2_j)+(Ep.T@Ep)/(2*(sigma2_j**2))
    #print(Ep)
    
    return np.concatenate(([g1],g2,[g3]))

# CMLE-hessian 
"""
Compute the Hessian matrix of the concentrated log-likelihood for a single response variable in the SAR model.

Output:
- Hessian matrix of shape (q0+2, q0+2)
"""
def hessian_initial_lj_0(nn, p, q, Wt, para, Yj, XX):
    
    q0 = 10
    
    rho_j = para[0]
    beta_j = para[1:(q0+1)]
    sigma2_j = para[-1]
    
    Sj = np.eye(nn)-rho_j*Wt
    S_inverse = np.linalg.inv(Sj)
    G = np.dot(Wt,S_inverse)
    WW = np.dot(Wt.T,Wt)
    WS = np.dot(Wt.T,Sj)
    SS = np.dot(Sj.T,Sj)
    Ep = Sj@Yj - XX@beta_j
    
    
    h11 = -np.trace(G@G)-np.dot(Yj.T@WW,Yj)/sigma2_j
    h12 = h21 = -(XX.T@(Wt@Yj)/sigma2_j).reshape(q0,1)
    h31 = h13 = -np.dot((Wt@Yj).T,Ep)/(sigma2_j**2)
    
    h22 = -XX.T@XX/sigma2_j
    h23 = h32 = -(XX.T@Ep/(2*sigma2_j**2)).reshape(q0,1)
    h33 = nn/(2*sigma2_j**2)-np.dot(Yj.T@SS,Yj)/(sigma2_j**3)
    
    H = np.block([
         [h11, h12.T, h13],
         [h21, h22, h23],
         [h31, h32.T, h33]
    ])
    
    return H

# CMLE-hessian -full q
"""
Compute the Hessian matrix of the concentrated log-likelihood for a single response variable in the SAR model.

Output:
- Hessian matrix of shape (q+2, q+2)
"""
def hessian_initial_lj(nn, p, q, Wt, para, Yj, XX):
    
  
    rho_j = para[0]
    beta_j = para[1:(q+1)]
    sigma2_j = para[-1]
    
    Sj = np.eye(nn)-rho_j*Wt
    S_inverse = np.linalg.inv(Sj)
    G = np.dot(Wt,S_inverse)
    WW = np.dot(Wt.T,Wt)
    WS = np.dot(Wt.T,Sj)
    SS = np.dot(Sj.T,Sj)
    Ep = Sj@Yj - XX@beta_j
    
    
    h11 = -np.trace(G@G)-np.dot(Yj.T@WW,Yj)/sigma2_j
    h12 = h21 = -(XX.T@(Wt@Yj)/sigma2_j).reshape(q,1)
    h31 = h13 = -np.dot((Wt@Yj).T,Ep)/(sigma2_j**2)
    
    h22 = -XX.T@XX/sigma2_j
    h23 = h32 = -(XX.T@Ep/(2*sigma2_j**2)).reshape(q,1)
    h33 = nn/(2*sigma2_j**2)-np.dot(Yj.T@SS,Yj)/(sigma2_j**3)
    
    H = np.block([
         [h11, h12.T, h13],
         [h21, h22, h23],
         [h31, h32.T, h33]
    ])
    
    return H

# Newton-CMLE
"""
Newton-Raphson optimization for concentrated likelihood in SAR model.

Output:
- Optimized parameter vector (q0+2,) and number of iterations
"""
def newton_sea_initial_0(nn, p, q, Wt, pa0, Yj, XX, max_iter = 50, eps = 1e-4):
    
    q0 = 10
    pa_new = pa0
    for t in range(max_iter):
        pa_pre = pa_new
        gradient = gardient_initial_lj_0(nn, p, q, Wt, pa_pre, Yj, XX)/nn 
        hessian =  hessian_initial_lj_0(nn, p, q, Wt, pa_pre, Yj, XX)/nn 
        diff = np.linalg.solve(hessian+0.001*np.eye(q0+2),gradient)
        pa_new = pa_pre - diff 
        if pa_new[-1]<0.01: pa_new[-1] = 0.01
        if pa_new[0]>1: pa_new[0] = 0.95
        #print(np.max(abs(diff)))
        if np.linalg.norm(diff) < eps:
            break
            
    return pa_new,t+1

# Newton-CMLE-full q
    """
Newton-Raphson optimization for concentrated likelihood in SAR model.

Output:
- Optimized parameter vector (q+2,) and number of iterations
"""
def newton_sea_initial(nn, p, q, Wt, pa0, Yj, XX, max_iter = 100, eps = 1e-4):
    
    pa_new = pa0
    for t in range(max_iter):
        pa_pre = pa_new
        gradient = gardient_initial_lj(nn, p, q, Wt, pa_pre, Yj, XX)/nn 
        hessian =  hessian_initial_lj(nn, p, q, Wt, pa_pre, Yj, XX)/nn 
        diff = np.linalg.solve(hessian+0.001*np.eye(q+2),gradient)
        pa_new = pa_pre - diff 
        if pa_new[-1]<0.1: pa_new[-1] = 0.1
        if pa_new[0]>1: pa_new[0] = 0.95
        # print(np.max(abs(diff)))
        if np.linalg.norm(diff) < eps:
            break
            
    return pa_new,t+1
    
# SCAD deriative function
"""
Compute the gradient of the SCAD (Smoothly Clipped Absolute Deviation) penalty 
for a given coefficient vector.

Parameters:
- beta_t: coefficient vector (can be scalar or array)
- lamba: regularization parameter λ
- a: SCAD parameter (default: 3.7)

Returns:
- grad: vector of SCAD derivative values, same shape as beta_t
"""
def SCAD_deriative_beta(beta_t, lamba, a = 3.7):
    
    abs_beta = np.abs(beta_t)
    grad = np.zeros_like(beta_t)
    
    mask1 = (abs_beta <= lamba)
    mask2 = (abs_beta > lamba) & (abs_beta <= a*lamba)
    
    grad[mask1] = lamba*np.sign(beta_t[mask1])
    grad[mask2] = ((a * lamba - abs_beta[mask2])/(a - 1))*np.sign(beta_t[mask2])
    
    return grad

# Newton-CMLE with SCAD
"""
Perform Newton-Raphson optimization for CMLE with SCAD penalty on beta coefficients.

Parameters:
- paj: initial parameter vector (rho, beta, sigma^2)
- lamba, a: SCAD regularization parameters
- XX, Yj, Wt: design matrix, response, and spatial weight matrix
- max_iter, eps: convergence control parameters

Returns:
- beta_new: estimated penalized beta coefficients
- number of iterations used
"""
def newton_sea_SCAD(nn, p, q, Wt, paj, Yj, XX, lamba, a=3.7, max_iter = 50, eps = 1e-3):
    
    rho_j = paj[0]
    beta_new = paj[1:(q+1)]
    sigma2_j = paj[-1]
    for t in range(max_iter):
        beta_pre = beta_new
        Ep = (np.eye(nn)-rho_j*Wt)@Yj - XX@beta_pre                    # Residual from transformed model
        gradient_beta = -(XX.T@Ep)
        hessian_beta = XX.T@XX
        grad_SCAD = SCAD_deriative_beta(beta_pre, lamba, a)            # SCAD penalty derivative
        S_lam_beta = np.diag(grad_SCAD)/abs(beta_pre)
        diff = np.linalg.inv(hessian_beta + nn*S_lam_beta)@(gradient_beta + nn*S_lam_beta@beta_pre)
        
        beta_new = beta_pre - diff                                     # Update step using penalized Newton direction
        #print(np.max(abs(diff)))
        if np.linalg.norm(diff) < eps:
            break
            
    return beta_new,t+1

"""
Compute the negative log-likelihood for the SAR model.

Parameters:
- rho_j: spatial autoregressive parameter
- beta_j: coefficient vector
- sigma2_j: error variance
- YY: response vector
- XX: design matrix
- Wt: spatial weight matrix

Returns:
- Negative log-likelihood value (scalar)
"""

def log_likelihood_sar(rho_j, beta_j, sigma2_j, YY, XX, Wt):
    
    nn = len(YY)
    qq = len(beta_j)
    
    A = np.eye(nn) - rho_j * Wt
    det_term = np.log(np.linalg.det(A))                     # log-determinant term for spatial correction
    residual = YY - rho_j * Wt @ YY - XX @ beta_j
    loglik = - np.log(2 * np.pi * sigma2_j)/2 + det_term/nn - (residual.T @ residual) / (2 * sigma2_j * nn)
    
    return -loglik                                          # Return negative log-likelihood for minimization