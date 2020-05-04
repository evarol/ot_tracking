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


def _transport_regression_poly(pts_1, pts_2, transport_mtx, degree):
    """Compute weighted polynomial regression using transport plan"""
    
    # Get pairs of points with values above threshold, and corresponding weights from P matrix
    idx_1, idx_2 = np.nonzero(transport_mtx)
    x = pts_1[idx_1]
    y = pts_2[idx_2]
    smp_wt = transport_mtx[idx_1, idx_2]

    # Use sklearn to minimize cost function
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
        ('linear', LinearRegression(fit_intercept=False))
    ])
    model.fit(x, y, linear__sample_weight=smp_wt)
   
    return model


def _em_registration(pts_1, pts_2, wts_1, wts_2, trans_fn, reg_fn, n_iter):
    """EM-based registration method using optimal transport plan."""
    
    model = [None] * n_iter
    t_mtx = [None] * n_iter
    t_log = [None] * n_iter
    
    pf_pts = [pts_1] + [None] * n_iter
    
    for i in range(n_iter):
        
        # E-step: Compute OT plan between current points and target points
        t_mtx[i], t_log[i] = trans_fn(pf_pts[i], pts_2, wts_1, wts_2)
        
        # M-step: Compute new mapping using transport plan
        model[i] =  reg_fn(pts_1, pts_2, t_mtx[i])
       
        # Update points 
        pf_pts[i + 1] = model[i].predict(pts_1)
       
    debug = {
        'model': model,
        'pf_pts': pf_pts,
        't_mtx': t_mtx,
        't_log': t_log,
    }
    
    return model[-1], debug

    
def ot_registration(pts_1, pts_2, wts_1, wts_2, degree, n_iter):
    """EM-based registration method using optimal transport plan."""
    
    poly_reg = partial(_transport_regression_poly, degree=degree) 
    
    return _em_registration(
        pts_1, pts_2, wts_1, wts_2, 
        trans_fn=_compute_ot, 
        reg_fn=poly_reg,
        n_iter=n_iter
    )


def gw_registration(pts_1, pts_2, wts_1, wts_2, degree, n_iter):
    """EM-based registration method using Gromov-Wasserstein transport plan."""
    
    poly_reg = partial(_transport_regression_poly, degree=degree) 
    
    return _em_registration(
        pts_1, pts_2, wts_1, wts_2, 
        trans_fn=_compute_gw, 
        reg_fn=poly_reg,
        n_iter=n_iter
    )


# def ot_registration(pts_1, pts_2, wts_1, wts_2, degree, n_iter):
#     """EM-based registration method using optimal transport plan."""
    
#     model = [None] * n_iter
#     t_mtx = [None] * n_iter
#     ot_log = [None] * n_iter
    
#     pf_pts = [pts_1] + [None] * n_iter
    
#     for i in range(n_iter):
        
#         # E-step: Compute OT plan between current points and target points
#         t_mtx[i], ot_log[i] = compute_ot(pf_pts[i], pts_2, wts_1, wts_2)
        
#         # M-step: Compute new mapping using transport plan
#         model[i] =  transport_regression_poly(pts_1, pts_2, t_mtx[i], degree)
       
#         # Update points 
#         pf_pts[i + 1] = model[i].predict(pts_1)
       
#     debug = {
#         'model': model,
#         'pf_pts': pf_pts,
#         't_mtx': t_mtx,
#         'ot_log': ot_log,
#     }
    
#     return model[-1], debug


# def gw_registration(pts_1, pts_2, wts_1, wts_2, degree, n_iter):
#     """EM-based registration method using Gromov-Wasserstein transport plan."""
    
#     model = [None] * n_iter
#     t_mtx = [None] * n_iter
#     gw_log = [None] * n_iter
    
#     pf_pts = [pts_1] + [None] * n_iter
    
#     for i in range(n_iter):
        
#         # E-step: Compute OT plan between current points and target points
#         t_mtx[i], gw_log[i] = compute_gw(pf_pts[i], pts_2, wts_1, wts_2)
        
#         # M-step: Compute new mapping using transport plan
#         model[i] =  transport_regression_poly(pts_1, pts_2, t_mtx[i], degree)
       
#         # Update points 
#         pf_pts[i + 1] = model[i].predict(pts_1)
       
#     debug = {
#         'model': model,
#         'pf_pts': pf_pts,
#         't_mtx': t_mtx,
#         'gw_log': gw_log,
#     }
 

## TODO: Delete this

# def ot_reg_linear_1(pts_1, pts_2, wts_1, wts_2):
#     """Use optimal transport to compute linear registration of two images.
    
#     Both images are represented as a list of spatial points with corresponding
#     non-negative weights (not necessarily normalized). These weights are 
#     normalized before computing the transport plan.
   
#     Args:
#         pts_1 (N*3 numpy.ndarray): Spatial points for 1st image
#         pts_2 (N*3 numpy.ndarray): Spatial points for 2nd image
#         wts_1 (N*0 numpy.ndarray): Weights for 1st image
#         wts_2 (N*0 numpy.ndarray): Weights for 2nd image
        
#     Returns:
#         3*0 numpy.ndarray: Intercept (alpha) of linear mapping
#         3*3 numpy.ndarray: Matrix (beta) of linear mapping
#         dict: Log object containing fields:
#             'ot_log' (dict): Log object for optimal transport computation
#             'P' (N*N numpy.ndarray): Optimal transport matrix 
    
#     """
    
#     # Normalize weights
#     p_1 = wts_1 / np.sum(wts_1)
#     p_2 = wts_2 / np.sum(wts_2)

#     # Normalized distance matrix 
#     M_nn = ot.dist(pts_1, pts_2, metric='sqeuclidean')
#     M = M_nn / np.median(M_nn)

#     # Compute transport plan
#     P, ot_log = ot.emd(p_1, p_2, M, log=True)

#     # Get pairs of points with values above threshold, and corresponding weights from P matrix
#     idx_1, idx_2 = np.nonzero(P)
#     x = pts_1[idx_1]
#     y = pts_2[idx_2]
#     smp_wt = P[idx_1, idx_2]

#     # Use sklearn.linear_model.LinearRegression to minimize cost function
#     model = LinearRegression(fit_intercept=True)
#     model.fit(x, y, sample_weight=smp_wt)

#     # Estimates of transform parameters
#     alpha = model.intercept_
#     beta = model.coef_
    
#     log = {
#         'ot': ot_log,
#         'P': P,
#     }

#     return alpha, beta, log


# def _compute_ot(pts_1, pts_2, wts_1, wts_2):
#     """Normalize weights and compute OT matrix."""
    
#     # Normalize weights
#     p_1 = wts_1 / np.sum(wts_1)
#     p_2 = wts_2 / np.sum(wts_2)

#     # Normalized distance matrix 
#     M_nn = ot.dist(pts_1, pts_2, metric='sqeuclidean')
#     M = M_nn / np.median(M_nn)
    
#     # Compute transport plan
#     return ot.emd(p_1, p_2, M, log=True)
    

# def _ot_reg_linear(pts_1, pts_2, wts_1, wts_2, p_mtx):
#     """Compute weighted linear regression using OT plan"""
    
#     # Get pairs of points with values above threshold, and corresponding weights from P matrix
#     idx_1, idx_2 = np.nonzero(p_mtx)
#     x = pts_1[idx_1]
#     y = pts_2[idx_2]
#     smp_wt = p_mtx[idx_1, idx_2]

#     # Use sklearn.linear_model.LinearRegression to minimize cost function
#     model = LinearRegression(fit_intercept=True)
#     model.fit(x, y, sample_weight=smp_wt)

#     # Estimates of transform parameters
#     alpha = model.intercept_
#     beta = model.coef_
    
#     return alpha, beta


# def ot_reg_linear_2(pts_s, pts_t, wts_s, wts_t, n_iter):
#     """EM-based OT registration method with linear model."""
    
#     alpha = [np.zeros(3)] + [None] * n_iter
#     beta = [np.eye(3)] + [None] * n_iter
    
#     pts = [None] * n_iter
#     p_mtx = [None] * n_iter
#     ot_log = [None] * n_iter
    
#     for i in range(n_iter):
        
#         # E-step: Compute OT between current points and target points
#         pts[i] = alpha[i] + pts_s @ beta[i].T
#         p_mtx[i], ot_log[i] = _compute_ot(pts[i], pts_t, wts_s, wts_t)
        
#         # M-step: Compute new mapping using transport plan
#         alpha[i + 1], beta[i + 1] =  _ot_reg_linear(pts_s, pts_t, wts_s, wts_t, p_mtx[i])
       
#     debug = {
#         'alpha': alpha,
#         'beta': beta,
#         'pts': pts,
#         'p_mtx': p_mtx,
#         'ot_log': ot_log,
#     }
    
#     return alpha[-1], beta[-1], debug


# def _ot_regression_poly(pts_1, pts_2, p_mtx, degree):
#     """Compute weighted polynomial regression using OT plan"""
    
#     # Get pairs of points with nonzero values and corresponding weights from P matrix
#     idx_1, idx_2 = np.nonzero(p_mtx)
#     x = pts_1[idx_1]
#     y = pts_2[idx_2]
#     smp_wt = p_mtx[idx_1, idx_2]

#     # Use sklearn to minimize cost function
#     model = Pipeline([
#         ('poly', PolynomialFeatures(degree=degree)),
#         ('linear', LinearRegression(fit_intercept=True))
#     ])
#     model.fit(x, y, linear__sample_weight=smp_wt)
   
#     return model


# def ot_reg_poly(pts_s, pts_t, wts_s, wts_t, n_iter, degree=3):
#     """EM-based OT registration method with polynomial model."""
    
#     model = [None] * n_iter
#     p_mtx = [None] * n_iter
#     ot_log = [None] * n_iter
    
#     pts = [pts_s] + [None] * n_iter
    
#     for i in range(n_iter):
        
#         # E-step: Compute OT between current points and target points
#         p_mtx[i], ot_log[i] = _compute_ot(pts[i], pts_t, wts_s, wts_t)
        
#         # M-step: Compute new mapping using transport plan
#         model[i] =  _ot_regression_poly(pts_s, pts_t, p_mtx[i], degree)
       
#         # Update points 
#         pts[i + 1] = model[i].predict(pts_s)
       
#     debug = {
#         'model': model,
#         'pts': pts,
#         'p_mtx': p_mtx,
#         'ot_log': ot_log,
#     }
    
#     return model[-1], debug


