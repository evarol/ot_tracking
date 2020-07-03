"""Optimal transport-based image registration methods"""

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from ot import emd
from ot.utils import dist 
from ot.gromov import gromov_wasserstein
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from otimage import deformations


def _normalized_dist_mtx(pts_1, pts_2, metric):
    """Return distance matrix normalized by median."""
    
    mtx_nn = dist(pts_1, pts_2, metric=metric)
    return mtx_nn / np.median(mtx_nn)


def _compute_ot(pts_1, pts_2, wts_1, wts_2):
    """Normalize weights and compute OT matrix."""
    
    # Normalize weights
    p_1 = wts_1 / np.sum(wts_1)
    p_2 = wts_2 / np.sum(wts_2)

    # Normalized distance matrix 
    c_mtx = _normalized_dist_mtx(pts_1, pts_2, 'sqeuclidean')
    
    # Compute transport plan
    return emd(p_1, p_2, c_mtx, log=True)


def _compute_gw(pts_1, pts_2, wts_1, wts_2):
    """Normalize weights and compute OT matrix."""
    
    # Normalize weights
    p_1 = wts_1 / np.sum(wts_1)
    p_2 = wts_2 / np.sum(wts_2)

    # Normalized distance matrices
    c_1 = _normalized_dist_mtx(pts_1, pts_1, metric='sqeuclidean')
    c_2 = _normalized_dist_mtx(pts_2, pts_2, metric='sqeuclidean')
    
    # Compute transport plan
    return gromov_wasserstein(c_1, c_2, p_1, p_2, 'square_loss', log=True)


def _transport_regression_affine(pts_1, pts_2, transport_mtx):
    """Compute weighted polynomial regression (degree=1) using transport plan"""
   
    # Get pairs of points with values above threshold, and corresponding weights from P matrix
    idx_1, idx_2 = np.nonzero(transport_mtx)
    x = pts_1[idx_1]
    y = pts_2[idx_2]
    smp_wt = transport_mtx[idx_1, idx_2]

    # Minimize cost function
    model = deformations.Affine()
    model.fit(x, y, smp_wt)
  
    return model


def _transport_regression_quadratic(pts_1, pts_2, transport_mtx):
    """Compute weighted polynomial regression (degree=2) using transport plan"""
   
    # Get pairs of points with values above threshold, and corresponding weights from P matrix
    idx_1, idx_2 = np.nonzero(transport_mtx)
    x = pts_1[idx_1]
    y = pts_2[idx_2]
    smp_wt = transport_mtx[idx_1, idx_2]

    # Minimize cost function
    model = deformations.Quadratic()
    model.fit(x, y, smp_wt)
  
    return model


def _transport_regression_cubic(pts_1, pts_2, transport_mtx):
    """Compute weighted polynomial regression (degree=3) using transport plan"""
   
    # Get pairs of points with values above threshold, and corresponding weights from P matrix
    idx_1, idx_2 = np.nonzero(transport_mtx)
    x = pts_1[idx_1]
    y = pts_2[idx_2]
    smp_wt = transport_mtx[idx_1, idx_2]

    # Minimize cost function
    model = deformations.Cubic()
    model.fit(x, y, smp_wt)
  
    return model


# TODO: Consider passing model instead of reg_fn
def _em_registration(mp_1, mp_2, trans_fn, reg_fn, n_iter):
    """EM-based registration method using optimal transport plan."""
    
    model = [None] * n_iter
    t_mtx = [None] * n_iter
    t_log = [None] * n_iter
    
    pf_pts = [mp_1.pts] + [None] * n_iter
    
    for i in range(n_iter):
        
        # E-step: Compute OT plan between current points and target points
        t_mtx[i], t_log[i] = trans_fn(pf_pts[i], mp_2.pts, mp_1.wts, mp_2.wts)
        
        # M-step: Compute new mapping using transport plan
        model[i] =  reg_fn(mp_1.pts, mp_2.pts, t_mtx[i])
       
        # Update points 
        pf_pts[i + 1] = model[i].predict(mp_1.pts)
       
    debug = {
        'model': model,
        'pf_pts': pf_pts,
        't_mtx': t_mtx,
        't_log': t_log,
    }
    
    return model[-1], debug

    
def ot_registration(mp_1, mp_2, degree, n_iter):
    """EM-based registration method using optimal transport plan."""
    
    if degree == 1:
        reg_fn = _transport_regression_affine
    elif degree == 2:
        reg_fn = _transport_regression_quadratic
    elif degree == 3:
        reg_fn = _transport_regression_cubic
    else:
        raise NotImplementedError
    
    return _em_registration(
        mp_1, mp_2,
        trans_fn=_compute_ot, 
        reg_fn=reg_fn,
        n_iter=n_iter
    )


def gw_registration(mp_1, mp_2, degree, n_iter):
    """EM-based registration method using Gromov-Wasserstein transport plan."""
    
    if degree == 1:
        reg_fn = _transport_regression_affine
    elif degree == 2:
        reg_fn = _transport_regression_quadratic
    elif degree == 3:
        reg_fn = _transport_regression_cubic
    else:
        raise NotImplementedError
    
    return _em_registration(
        mp_1, mp_2,
        trans_fn=_compute_gw, 
        reg_fn=reg_fn,
        n_iter=n_iter
    )