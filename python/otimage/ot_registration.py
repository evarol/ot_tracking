# -*- coding: utf-8 -*-
"""Optimal transport (OT) registration methods used for dNMF"""

# import ot
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# from otimage import imagerep

from functools import partial

import numpy as np
from ot import emd
from ot.utils import dist 
from ot.gromov import gromov_wasserstein
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def compute_ot_ent(pts_1, pts_2, wts_1, wts_2, lambd):
    """Normalize weights and compute OT matrix."""
    # Normalize weights
    p_1 = wts_1 / np.sum(wts_1)
    p_2 = wts_2 / np.sum(wts_2)
    
    # Normalized distance matrix 
    M = 1-np.exp(-ot.dist(pts_1, pts_2, metric='euclidean')/100)
#    M = M_nn / np.median(M_nn)
    
    M = np.append(M,.99*np.ones((1,M.shape[1])),0)
    
    p_1 = p_1*.2
    p_1 = np.append(p_1,.8)
    
    # Compute transport plan
    emd = ot.bregman.sinkhorn_knopp(p_1, p_2, M, lambd, log=True)
    return emd


def normalized_dist_mtx(pts_1, pts_2, metric):
    """Return distance matrix normalized by median."""
    
    mtx_nn = dist(pts_1, pts_2, metric=metric)
    return mtx_nn / np.median(mtx_nn)


def compute_ot(pts_1, pts_2, wts_1, wts_2):
    """Normalize weights and compute OT matrix."""
    
    # Normalize weights
    p_1 = wts_1 / np.sum(wts_1)
    p_2 = wts_2 / np.sum(wts_2)

    # Normalized distance matrix 
    c_mtx = normalized_dist_mtx(pts_1, pts_2, 'sqeuclidean')
    
    # Compute transport plan
    return emd(p_1, p_2, c_mtx, log=True)


def compute_gw(pts_1, pts_2, wts_1, wts_2):
    """Normalize weights and compute OT matrix."""
    
    # Normalize weights
    p_1 = wts_1 / np.sum(wts_1)
    p_2 = wts_2 / np.sum(wts_2)

    # Normalized distance matrices
    c_1 = normalized_dist_mtx(pts_1, pts_1, metric='sqeuclidean')
    c_2 = normalized_dist_mtx(pts_2, pts_2, metric='sqeuclidean')
    
    # Compute transport plan
    return gromov_wasserstein(c_1, c_2, p_1, p_2, 'square_loss', log=True)


def transport_regression_poly(pts_1, pts_2, transport_mtx, degree):
    """Compute weighted polynomial regression using transport plan"""
    
    # Get pairs of points with values above threshold, and corresponding weights from P matrix
    idx_1, idx_2 = np.nonzero(transport_mtx)
    x = pts_1[idx_1]
    y = pts_2[idx_2]
    smp_wt = transport_mtx[idx_1, idx_2]

    # Use sklearn to minimize cost function
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression(fit_intercept=True))
    ])
    model.fit(x, y, linear__sample_weight=smp_wt)
   
    return model


def ot_registration(pts_1, pts_2, wts_1, wts_2, degree, n_iter):
    """EM-based registration method using optimal transport plan."""
    
    model = [None] * n_iter
    t_mtx = [None] * n_iter
    ot_log = [None] * n_iter
    
    pf_pts = [pts_1] + [None] * n_iter
    
    for i in range(n_iter):
        
        # E-step: Compute OT plan between current points and target points
        t_mtx[i], ot_log[i] = compute_ot(pf_pts[i], pts_2, wts_1, wts_2)
        
        # M-step: Compute new mapping using transport plan
        model[i] =  transport_regression_poly(pts_1, pts_2, t_mtx[i], degree)
       
        # Update points 
        pf_pts[i + 1] = model[i].predict(pts_1)
       
    debug = {
        'model': model,
        'pf_pts': pf_pts,
        't_mtx': t_mtx,
        'ot_log': ot_log,
    }
    
    return model[-1], debug


def gw_registration(pts_1, pts_2, wts_1, wts_2, degree, n_iter):
    """EM-based registration method using Gromov-Wasserstein transport plan."""
    
    model = [None] * n_iter
    t_mtx = [None] * n_iter
    gw_log = [None] * n_iter
    
    pf_pts = [pts_1] + [None] * n_iter
    
    for i in range(n_iter):
        
        # E-step: Compute OT plan between current points and target points
        t_mtx[i], gw_log[i] = compute_gw(pf_pts[i], pts_2, wts_1, wts_2)
        
        # M-step: Compute new mapping using transport plan
        model[i] =  transport_regression_poly(pts_1, pts_2, t_mtx[i], degree)
       
        # Update points 
        pf_pts[i + 1] = model[i].predict(pts_1)
       
    debug = {
        'model': model,
        'pf_pts': pf_pts,
        't_mtx': t_mtx,
        'gw_log': gw_log,
    }
    
    return model[-1], debug


def initialize_beta(pts, wts, frame, n_iter=5, mp_cov=[15,15,15], mp_k=60):
    
    # Extract MP components from frame
    pts_t, wts_t, _ = imagerep.mp_gaussian(frame, np.diag(mp_cov), mp_k)
    
    # Eliminate negative and zero-valued weights (is this an actual problem?)
    wts[wts <= 0] = 1e-5
    wts_t[wts_t <= 0] = 1e-5
    
    # Fit model for spatial map between frames
    model, _ = ot_registration(
        pts, pts_t, wts.squeeze(), wts_t.squeeze(), degree=1, n_iter=n_iter)
    
    # Apply spatial map to points
    pts_transformed = model.predict(pts)
    
    #plt.imshow(frame.max(2)[0])
    #plt.scatter(positions[:,1],positions[:,0],color='r')
    #plt.scatter(positions_t[:,1],positions_t[:,0],color='b')
    #plt.scatter(positions_transformed[:,1],positions_transformed[:,0],color='g')
    #plt.legend(['source', 'target', 'source_{tr}'])
    #plt.show()
    #plt.pause(1)
    
    # Extract beta from model parameters
    beta = np.zeros((10,3))
    beta[0,:] = model.named_steps['linear'].intercept_
    beta[1:4,:] = model.named_steps['linear'].coef_[:,1:].T
    
    #print(positions@model.named_steps['linear'].coef_[:,1:].T+model.named_steps['linear'].intercept_)
    #print(positions_transformed)
    print(beta)
    
    return beta, pts_transformed