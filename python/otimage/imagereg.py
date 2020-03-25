"""Optimal transport-based image registration methods"""

import numpy as np
import matplotlib.pyplot as plt
import ot
from sklearn.linear_model import LinearRegression


def ot_reg_linear(pts_1, pts_2, wts_1, wts_2):
    """Use optimal transport to compute linear registration of two images.
    
    Both images are represented as a list of spatial points with corresponding
    non-negative weights (not necessarily normalized). These weights are 
    normalized before computing the transport plan.
   
    Args:
        pts_1 (N*3 numpy.ndarray): Spatial points for 1st image
        pts_2 (N*3 numpy.ndarray): Spatial points for 2nd image
        wts_1 (N*0 numpy.ndarray): Weights for 1st image
        wts_2 (N*0 numpy.ndarray): Weights for 2nd image
        
    Returns:
        3*0 numpy.ndarray: Intercept (alpha) of linear mapping
        3*3 numpy.ndarray: Matrix (beta) of linear mapping
        dict: Log object containing fields:
            'ot_log' (dict): Log object for optimal transport computation
            'P' (N*N numpy.ndarray): Optimal transport matrix 
    
    """
    
    # Normalize weights
    p_1 = wts_1 / np.sum(wts_1)
    p_2 = wts_2 / np.sum(wts_2)

    # Normalized distance matrix 
    M_nn = ot.dist(pts_1, pts_2, metric='sqeuclidean')
    M = M_nn / np.median(M_nn)

    # Compute transport plan
    P, ot_log = ot.emd(p_1, p_2, M, log=True)

    # Get pairs of points with values above threshold, and corresponding weights from P matrix
    idx_1, idx_2 = np.nonzero(P)
    x = pts_1[idx_1]
    y = pts_2[idx_2]
    smp_wt = P[idx_1, idx_2]

    # Use sklearn.linear_model.LinearRegression to minimize cost function
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y, sample_weight=smp_wt)

    # Estimates of transform parameters
    alpha = model.intercept_
    beta = model.coef_
    
    log = {
        'ot': ot_log,
        'P': P,
    }

    return alpha, beta, log


#def _compute_ot(pts_1, pts_2, wts_1, wts_2):
    
    ## Normalize weights
    #p_1 = wts_1 / np.sum(wts_1)
    #p_2 = wts_2 / np.sum(wts_2)

    ## Normalized distance matrix 
    #M_nn = ot.dist(pts_1, pts_2, metric='sqeuclidean')
    #M = M_nn / np.median(M_nn)
    
    ## Compute transport plan
    #return ot.emd(p_1, p_2, M, log=True)
    
    
# TODO: Consider changing name here
#def _ot_reg(pts_1, pts_2, wts_1, wts_2, p_mtx):
    
    ## Get pairs of points with values above threshold, and corresponding weights from P matrix
    #idx_1, idx_2 = np.nonzero(p_mtx)
    #x = pts_1[idx_1]
    #y = pts_2[idx_2]
    #smp_wt = p_mtx[idx_1, idx_2]

    ## Use sklearn.linear_model.LinearRegression to minimize cost function
    #model = LinearRegression(fit_intercept=True)
    #model.fit(x, y, sample_weight=smp_wt)

    ## Estimates of transform parameters
    #alpha = model.intercept_
    #beta = model.coef_
    
    #return alpha, beta

# TODO: Update docstring
#def ot_reg_linear(pts_s, pts_t, wts_s, wts_t, n_iter):
    #"""Use optimal transport to compute linear registration of two images.
    
    #Both images are represented as a list of spatial points with corresponding
    #non-negative weights (not necessarily normalized). These weights are 
    #normalized before computing the transport plan.
   
    #Args:
    #    pts_1 (N*3 numpy.ndarray): Spatial points for 1st image
    #    pts_2 (N*3 numpy.ndarray): Spatial points for 2nd image
    #    wts_1 (N*0 numpy.ndarray): Weights for 1st image
    #    wts_2 (N*0 numpy.ndarray): Weights for 2nd image
        
    #Returns:
    #    3*0 numpy.ndarray: Intercept (alpha) of linear mapping
    #    3*3 numpy.ndarray: Matrix (beta) of linear mapping
    #    dict: Debug information. Containins fields:
    #        'ot_log' (dict): Log object for optimal transport computation
    #        'P' (N*N numpy.ndarray): Optimal transport matrix 
    
    #"""
 
    
    #alpha = [np.zeros(3)] + [None] * n_iter
    #beta = [np.eye(3)] + [None] * n_iter
    
    #pts = [None] * n_iter
    #p_mtx = [None] * n_iter
    #ot_log = [None] * n_iter
    
    #for i in range(n_iter):
        
        ## E-step: Compute OT between current points and target points
        #pts[i] = alpha[i] + pts_s @ beta[i].T
        #p_mtx[i], ot_log[i] = _compute_ot(pts[i], pts_t, wts_s, wts_t)
        
        ## M-step: Compute new mapping using transport plan
        #alpha[i + 1], beta[i + 1] =  _ot_reg(pts_s, pts_t, wts_s, wts_t, p_mtx[i])
       
    #debug = {
    #    'alpha': alpha,
    #    'beta': beta,
    #    'pts': pts,
    #    'p_mtx': p_mtx,
    #    'ot_log': ot_log,
    #}
    
    #return alpha[-1], beta[-1], debug
