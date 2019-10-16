"""Utility functions for notebooks"""

import numpy as np
import ot

def wasserstein_interp_2d(img_1, img_2, reg, alpha):
    """Compute Wasserstein interpolation between two images.
    
    Args:
        img_1 (numpy.ndarray): First image for interpolation. Entries must sum
            to one.
        img_2 (numpy.ndarray): Second image for interpolation. Must have same
            dimensions as `img_` and entries must sum to one.
        reg (float): Regularization parameter. Must be positive.
        alpha (float): Interpolation parameter. Must be between 0 and 1.
        
    Returns:
        numpy.ndarray: Interpolated image (same dimensions as `img_1` and
            `img_2`)
    
    """
    
    # Get dimensions of data
    nx_1, ny_1 = img_1.shape
    nx_2, ny_2 = img_2.shape
    assert(nx_1 == nx_2)
    assert(ny_1 == ny_2)
    nx, ny = (nx_1, ny_1)
    
    # Compute distance matrix between all grid points
    [xg, yg] = np.meshgrid(np.arange(nx), np.arange(ny))
    grid_vals = np.hstack((xg.reshape(-1, 1), yg.reshape(-1, 1)))
    M = ot.utils.dist(grid_vals)

    # Normalize distance matrix -- this prevents conditioning issues
    M = M / np.median(M)

    # Compute barycenter between images to use as interpolant
    A = np.hstack((img_1.reshape(-1, 1), img_2.reshape(-1, 1)))
    weights = np.array([1 - alpha, alpha])
    interp_vec = ot.bregman.barycenter(A, M, reg, weights)
    
    return interp_vec.reshape((nx, ny))