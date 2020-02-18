"""Optimal transport-based image registration methods"""

import numpy as np
import matplotlib.pyplot as plt
import ot
from sklearn.linear_model import LinearRegression


def ot_reg_linear(pts_1, pts_2, wts_1, wts_2):
    """Use optimal transport to compute linear registration of two images.
    
    Both images are represented as a list of spatial points with corresponding
    non-negative weights (not necessarily normalized). To compute the optimal
    transport plan, we concatenate the lists of spatial points to create a
    shared coordinate space that can be used to represent both images, and
    normalize the weights to make them probability distributions.
    
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
    
    # Matrix containing locations of both sets of MPs
    pts = np.concatenate((pts_1, pts_2), axis=0)

    # Put weights in common coordinate system and normalize them
    p_1_nn = np.concatenate((wts_1, np.zeros(wts_2.shape[0])), axis=0)
    p_2_nn = np.concatenate((np.zeros(wts_1.shape[0]), wts_2), axis=0)
    p_1 = p_1_nn / np.sum(p_1_nn)
    p_2 = p_2_nn / np.sum(p_2_nn)

    # Normalized distance matrix 
    M_nn = ot.dist(pts, metric='sqeuclidean')
    M = M_nn / np.median(M_nn)

    # Compute transport plan
    P, ot_log = ot.emd(p_1, p_2, M, log=True)

    # Get pairs of points with values above threshold, and corresponding weights from P matrix
    idx_1, idx_2 = np.nonzero(P)
    x = pts[idx_1]
    y = pts[idx_2]
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

