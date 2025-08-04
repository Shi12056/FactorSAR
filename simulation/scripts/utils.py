'''
This is a python file containing all functions needed in the parameter estimation part of simulation studies.
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
    
    q0 = 10
    Omega = rng.multivariate_normal(mean, cov, (nn,), 'raise')   
    X = rng.normal(0,1,(nn,q0))
    Z = rng.normal(0,1,(nn,d))
    
    Y = np.zeros((nn,p))
    for j in range(p):
        # Compute spatially lagged response via inverse of (I - rho * W)
        y_j = np.dot(np.linalg.inv(np.eye(nn)-rho[j]*W), X@beta[j,:q0] + Z@bc[j,:]+Omega[:,j])
        Y[:,j] = y_j 
        
    return Y, X, Z

# CMLE-gradient
"""
Compute the gradient of the concentrated log-likelihood for a single response variable in the SAR model.

Output:
- Gradient vector of length (q0 + 2)
"""
def gardient_initial_lj(nn, p, q, WW, para, Yj, XX):
    
    q0 = 10
    
    rho_j = para[0]
    beta_j = para[1:(q0+1)]
    sigma2_j = para[-1]
    
    
    Sj = np.eye(nn)-rho_j*WW                           # Spatial transformation matrix
    S_inverse = np.linalg.inv(Sj)                      # Inverse matrix
    G = WW@S_inverse                                   # Used in trace term
    Ep = Sj@Yj - XX@beta_j                             # Residuals
    
    g1 = -np.trace(G)+np.dot((WW@Yj).T,Ep)/sigma2_j    # dL/drho
    g2 = (XX.T@Ep)/sigma2_j                            # dL/dbeta
    g3 = -nn/(2*sigma2_j)+(Ep.T@Ep)/(2*(sigma2_j**2))  # dL/dsigma^2
    #print(Ep)
    
    return np.concatenate(([g1],g2,[g3]))

# CMLE-hessian
"""
Compute the Hessian matrix of the concentrated log-likelihood for a single response variable in the SAR model.

Output:
- Hessian matrix of shape (q0+2, q0+2)
"""
def hessian_initial_lj(nn, p, q, Wt, para, Yj, XX):
    
    q0 = 10
    
    rho_j = para[0]
    beta_j = para[1:(q0+1)]
    sigma2_j = para[-1]
    
    Sj = np.eye(nn)-rho_j*Wt                                # Spatial transformation matrix
    S_inverse = np.linalg.inv(Sj)                           # Inverse matrix
    G = np.dot(Wt,S_inverse)                                # Spatial multiplier
    WW = np.dot(Wt.T,Wt)                                    # Used in rho block
    WS = np.dot(Wt.T,Sj)
    SS = np.dot(Sj.T,Sj)
    Ep = Sj@Yj - XX@beta_j                                  # Residuals
    
    
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

# Newton--CMLE
"""
Newton-Raphson optimization for concentrated likelihood in SAR model.

Output:
- Optimized parameter vector (q0+2,) and number of iterations
"""
def newton_sea_initial(nn, p, q, Wt, pa0, Yj, XX, max_iter = 50, eps = 1e-4):
    
    q0 = 10
    pa_new = pa0
    for t in range(max_iter):
        pa_pre = pa_new
        gradient = gardient_initial_lj(nn, p, q, Wt, pa_pre, Yj, XX)/nn 
        hessian =  hessian_initial_lj(nn, p, q, Wt, pa_pre, Yj, XX)/nn 
        diff = np.linalg.solve(hessian+0.001*np.eye(q0+2),gradient)          # Regularized Newton step
        pa_new = pa_pre - diff 
        if pa_new[-1]<0.01: pa_new[-1] = 0.01
        if pa_new[0]>1: pa_new[0] = 0.95
        #print(np.max(abs(diff)))
        if np.linalg.norm(diff) < eps:                                       # Convergence check
            break
            
    return pa_new,t+1
    
# SAR optimization function
"""
Apply CMLE to all p response dimensions by calling Newton optimization.

Output:
- Estimated rho: (p,), beta: (q0, p), sigma^2: (p,), and iteration counts
"""
def optimize_initial(nn, p, q, Wt,YY, XX, bb, save_path=None):
    
    q0 = 10
    para = np.zeros((q0+2,p))
    ite = np.zeros(p)
    par = np.zeros(q0+2)
    par[-1] = 0.1
    for j in range(p):
        res = newton_sea_initial(nn, p, q, Wt, par, YY[:,j], XX)             # Estimate parameters for j-th response
        para[:,j] = res[0]
        if para[0,j]> 0.95: 
            para[0,j] = 0.95
        if para[0,j]< 0.05: 
            para[0,j] = 0.05
        ite[j] = res[1]
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'a') as f1:
                f1.write(str(bb)+', initial: '+ str(j) + ', '+ str(para[0,j]) +'\n')
        #print('initial:',j, para[0,j])
    
    return para[0,:],para[1:(q0+1),:], para[-1,:],ite

# FMLE-gradient
"""
Compute the gradient of the full log-likelihood for a single response  in the FMLE setting.

Output:
- Gradient vector of length (q0 + d + 2)
"""
def gardient_lj(nn, p, q, d, Wt,the0, Z_ee, Yj, XX):
    
    q0 = 10
    rhoj = the0[0]
    betaj = the0[1:(q0+1)]
    bj = the0[(q0+1):(q0+d+1)]
    tauj2 = the0[-1]
    
    Sj = np.eye(nn)-rhoj*Wt                         # Spatial transformation
    S_inverse = np.linalg.inv(Sj)
    G = Wt@S_inverse
    Ep = Sj@Yj - XX@betaj - Z_ee@bj                 # Residuals
    
    g1 = -np.trace(G)+np.dot((Wt@Yj).T,Ep)/tauj2
    g2 = (XX.T@Ep)/tauj2
    g3 = (Z_ee.T@Ep)/tauj2
    g4 = -nn/(2*tauj2)+(Ep.T@Ep)/(2*tauj2**2)
    
    return np.concatenate(([g1],g2,g3,[g4]))

# FMLE-hessian
"""
Compute the Hessian matrix of the full log-likelihood for a single response in the FMLE setting.

Output:
- Hessian matrix of shape (q0 + d + 2, q0 + d + 2)
"""
def hessian_lj(nn, p, q, d, Wt, the0, Z_e, Yj, XX):
    
    q0 = 10
    rhoj = the0[0]
    betaj = the0[1:(q0+1)]
    bj = the0[(q0+1):(q0+d+1)]
    tauj2 = the0[-1]
    
    Sj = np.eye(nn)-rhoj*Wt
    S_inverse = np.linalg.inv(Sj)
    G = np.dot(Wt,S_inverse)
    WY = Wt@Yj
    Ep = Sj@Yj - XX@betaj - Z_e@bj                 # Residuals
    
    h11 = -np.trace(G@G) - WY.T@WY/tauj2
    
    h12 = h21 = -(XX.T@WY/tauj2).reshape(q0,1)
    h13 = h31 = -(Z_e.T@WY/tauj2).reshape(d,1)
    h14 = h41 = -WY.T@Ep/tauj2**2
    
    h22 = -XX.T@XX/tauj2
    h23 = h32 = -XX.T@Z_e/tauj2
    h24 = h42 = -(XX.T@Ep/(2*tauj2**2)).reshape(q0,1)
    
    h33 = -Z_e.T@Z_e/tauj2
    h34 = h43 = -(Z_e.T@Ep/(2*tauj2**2)).reshape(d,1)
    
    h44 = nn/(2*tauj2**2)-Ep.T@Ep/(tauj2**3)
    
    H = np.block([
         [h11, h12.T, h13.T, h14],
         [h21, h22, h23, h24],
         [h31, h32.T, h33, h34],
         [h41, h42.T, h43.T, h44]
    ])
    
    return H


# Newton--FMLE
"""
Newton-Raphson optimization for full likelihood estimation in FMLE setting.

Output:
- Estimated parameter vector of length (q0 + d + 2) and iteration count
"""
def newton_sea(nn, p, q, d, Wt, theta0, Z_e, Yj, Xt, max_iter = 50, eps = 1e-4):
    
    q0 = 10
    theta_new = theta0
    for t in range(max_iter):
        theta_pre = theta_new
        gradient = gardient_lj(nn, p,q,d,Wt, theta_pre, Z_e, Yj,Xt)/nn 
        hessian =  hessian_lj(nn, p,q,d,Wt, theta_pre, Z_e, Yj,Xt)/nn 
        diff = np.linalg.solve(hessian+0.001*np.eye(q0+d+2),gradient)
        theta_new = theta_pre - diff
        if theta_new[-1]<0.01: theta_new[-1] = 0.01
        if theta_new[0]>1: theta_new[0] = 0.95

        if np.linalg.norm(diff) < eps:
            break
            
    return theta_new,t+1

# FMLE optimization function
"""
Full Maximum Likelihood Estimation (FMLE) for all p response variables.

Returns:
- rho estimates, beta estimates, b estimates, tau^2 estimates, and iteration count per response
"""
def optimize(nn, p, q, d, Wt, Z_e, Yt, Xt, bb, save_path=None):
    
    q0 = 10
    g = q0+d+2
    theta = np.zeros((g,p))
    itee = np.zeros(p)
    theta0 = np.zeros(g)
    theta0[-1] = 0.1
    for j in range(p):
        res = newton_sea(nn, p, q, d, Wt, theta0, Z_e, Yt[:,j], Xt)
        theta[:,j] = res[0]
        itee[j] = res[1]
        if theta[0,j]> 0.95: 
            theta[0,j] = 0.95
        if theta[0,j]< 0.05: 
            theta[0,j] = 0.05
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'a') as f1:
                f1.write(str(bb)+', update: '+ str(j) + ', '+ str(theta[0,j]) +'\n')
    
    return theta[0,:],theta[1:(q0+1),:],theta[(q0+1):(q0+d+1),:], theta[-1,:], itee

# Factor estimation function
"""
Estimate latent factors Z_e and projection matrix M.

Returns:
- Z_e: estimated latent scores for current data
- M: estimated loading matrix for latent space
"""
def Z_est(nn, p, q, d, Wt0, Wt1, Y_1, X_1, rho_e, beta_e, Yb, Xb, rho_hb, beta_hb):
    
    Eb = Yb - (Wt0@Yb)@np.diag(rho_hb) - Xb@beta_hb
    Sigm = Eb.T@Eb/nn 
    U, G, V = np.linalg.svd(Sigm)
    M = U[:,:d]@np.diag(G[:d])**0.5
    
    E = Y_1 - (Wt1@Y_1)@np.diag(rho_e) - X_1@beta_e
     
    Z_e = E@M/p
    
    return Z_e, M

# Confidence interval for FMLE
"""
Estimate the confidence interval for the spatial autocorrelation parameter (rho_j) using the sandwich variance estimator, suitable for Feasible Maximum Likelihood Estimation (FMLE) in spatial econometric models.

Parameters:
- j: index of the response variable
- All other arguments: required for computing the full variance decomposition

Returns:
- lower, upper: bounds of confidence interval for rho_j
"""
def CI_est_test(j,nn, p, q0, d, WW, rho_est, beta_est, b_est, tau2_est, B_hat, sigma2_hat, YY, XX, Z_est, z_alpha):
    
    # ticn1 = time()
    # Construct transformation matrix (I - rho * W)
    S = np.eye(nn)-rho_est[j]*WW
    S_inverse = np.linalg.inv(S)
    G = np.dot(WW,S_inverse)
    Gs = G + G.T
    GXbe = G@XX@beta_est[:,j]
    GZb = G@Z_est@b_est[:,j]

    c1 = tau2_est[j]*np.trace(G@Gs)
    c21 = GXbe + GZb
    C11 = c1 + c21.T@c21
    C12 = C21 = (XX.T@GXbe).reshape(q0,1)
    C13 = C31 = (Z_est.T@GZb).reshape(d,1)
    C14 = C41 = np.trace(G)
    C22 = XX.T@XX
    C23 = C32 = XX.T@Z_est
    C24 = C42 = np.zeros((q0,1))
    C33 = Z_est.T@Z_est
    C34 = C43 = np.zeros((d,1))
    C44 = nn/(2*tau2_est[j])
    # Assemble the block matrix of the expected Fisher information        
    Sigma2 = np.block([
             [C11, C12.T, C13.T, C14],
             [C21, C22, C23, C24],
             [C31, C32.T, C33, C34],
             [C41, C42.T, C43.T, C44],
        ])/(nn*tau2_est[j])
    
    Sigma2_inv = np.linalg.inv(Sigma2)
    # tocn1 = time()
    # print(tocn1 - ticn1) 
    
    g = q0+d+2
    Mbtj = B_hat@b_est[:,j]
    Omega_hat = S@YY[:,j] - XX@beta_est[:,j] - Z_est@b_est[:,j]
    # Initialize cumulative matrices for sandwich variance terms
    Aj = np.zeros((nn*d,g*nn*d))
    Dj = np.zeros((nn*d,g))
    # Loop over all response variables to compute the variability due to estimation
    for k in range(p):
        Sk = np.eye(nn)-rho_est[k]*WW
        S_inverse = np.linalg.inv(Sk)
        G = np.dot(WW,S_inverse)
        GXbeta = G@XX@beta_est[:,k]
        
        C11 = np.trace(G@(G+G.T)) + GXbeta.T@GXbeta/sigma2_hat[k]
        C12 = C21 = (XX.T@GXbeta/sigma2_hat[k]).reshape(q0,1)
        C13 = C31 = np.trace(G)/sigma2_hat[k]
        C22 = XX.T@XX/sigma2_hat[k]
        C23 = C32 = np.zeros((q0,1))
        C33 = nn/(2*sigma2_hat[k]**2)
        
        Sigma = np.block([
             [C11, C12.T, C13],
             [C21, C22, C23],
             [C31, C32.T,C33]
        ])/nn
        H = np.linalg.inv(Sigma)
        H_rho = H[0,:]
        H_beta = H[1:-1,:]

        # Derivative matrices A1, A2, A3 used
        A2 = G.T/sigma2_hat[k]
        A3 = np.eye(nn)/(2*sigma2_hat[k]**2)
        A1 = H_rho[0]*A2 + H_rho[-1]*A3
        
        # Score and cross-derivative terms for Delta-method-based correction
        cjk_rho = YY[:,j].T@WW.T@WW@YY[:,k]*Mbtj[k]/nn
        cjk_beta = XX.T@WW@YY[:,k]*Mbtj[k]/nn
        cjk_b = Z_est.T@WW@YY[:,k]*Mbtj[k]/nn - B_hat[k,:]*(Omega_hat@WW@YY[:,k])/nn
        cjk_tau2 = Omega_hat@WW@YY[:,k]*Mbtj[k]/(nn*tau2_est[j])
        Qjk_rho = (XX.T@WW@YY[:,j]*Mbtj[k]/nn).reshape(q0,1)
        Qjk_beta = XX.T@XX*Mbtj[k]/nn # q0 c q0
        Qjk_b = XX.T@Z_est*Mbtj[k]/nn - (XX.T@Omega_hat).reshape(q0,1)@B_hat[k,:].reshape(1,d)/nn # q0 c d
        Qjk_tau2 = (XX.T@Omega_hat*Mbtj[k]/(nn*tau2_est[j])).reshape(q0,1)
        # Concatenate all vectors for variance propagation
        cjk_Theta = np.hstack([cjk_rho, cjk_beta, cjk_b, cjk_tau2]) #  g
        Qjk_Theta = np.concatenate([Qjk_rho, Qjk_beta, Qjk_b, Qjk_tau2], axis=1) # q0 c g
        
        # Compute correction matrices Ak and Djk
        vjk1 = cjk_Theta # g 
        vjk2 = Qjk_Theta.T@H_beta[:,0] # g 
        vjk3 = Qjk_Theta.T@H_beta[:,-1] # g 
        
        Ak = np.kron(A1, vjk1.reshape(1,g)) + np.kron(A2, vjk2.reshape(1,g)) + np.kron(A1, vjk3.reshape(1,g))
        Akb = np.kron(b_est[:,k].reshape(d,1)@b_est[:,k].reshape(1,d), Ak)
        Aj = Aj + Akb/p
    
        vjk = (cjk_Theta*H_rho[0] + Qjk_Theta.T@H_beta[:,0])/sigma2_hat[k] # g
        Ujk = (cjk_Theta.reshape(g,1)@H_beta[:,0].reshape(1,q0) + Qjk_Theta.T@H_beta[:,1:-1])/sigma2_hat[k] # g c q0
        Djk = GXbeta.reshape(nn,1)@vjk.reshape(1,g) + XX@Ujk.T # n c g
        Djkb = np.kron(b_est[:,k].reshape(d,1), Djk)
        Dj = Dj + Djkb/p

    # tocn1 = time()
    # print(tocn1 - ticn1)
    # Compute the covariance matrix of the correction term using higher-order moments
    Sigma_Q = np.zeros((g, g))
    H_hat = B_hat.T @ B_hat / p
    HI = np.kron(H_hat @ H_hat.T, np.eye(nn))
    Z_hat = Z_est @ np.linalg.inv(H_hat)
    mu3z = np.mean(Z_hat**3)
    mu4z = np.mean(Z_hat**4)
    I_nd = np.eye(nn * d)
    D_matrix = Dj @ np.eye(g)  
    for i1 in range(g):
        Dbej1 = D_matrix[:, i1]
        Abej1 = Aj @ np.kron(I_nd, np.eye(g)[:, i1].reshape(g, 1))  
        diag_Abej1 = np.diag(Abej1)

        for i2 in range(g):
            # ticnn1 = time()
            Dbej2 = D_matrix[:, i2]
            Abej2 = Aj @ np.kron(I_nd, np.eye(g)[:, i2].reshape(g, 1))
            diag_Abej2 = np.diag(Abej2)

            VQ1 = Dbej1.T @ HI @ Dbej2
            VQ2 = np.trace(Abej1 @ HI @ (Abej2 + Abej2.T) @ HI)
            VQ3 = mu3z * (np.sum(diag_Abej1 * Dbej2) + np.sum(diag_Abej2 * Dbej1))
            VQ4 = mu4z * np.sum(diag_Abej1 * diag_Abej2)

            Sigma_Q[i1, i2] = (VQ1 + VQ2 + VQ3 + VQ4) / (nn * tau2_est[j]**2)
            # tocnn1 = time()
            # print(tocnn1 - ticnn1)

    # tocn1 = time()
    # print(tocn1 - ticn1) 
    # print(Sigma_Q) 
    # Final robust variance estimator using sandwich formula
    Sigma_Q2 = Sigma2_inv@Sigma_Q@Sigma2_inv
    sig = Sigma2_inv[0,0] + Sigma_Q2[0,0]

    # Confidence interval bounds for rho_j
    lower = rho_est[j] - z_alpha*sig**0.5/nn**0.5
    upper = rho_est[j] + z_alpha*sig**0.5/nn**0.5

    
    return lower,upper