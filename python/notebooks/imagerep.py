"""Functions for creating low-dimensional image representations"""

import numpy as np
from scipy import ndimage
from scipy.stats import multivariate_normal
from skimage.feature import peak_local_max
from skimage.util import img_as_float, img_as_uint
from skimage.measure import label
from skimage.filters import threshold_otsu
from skimage.segmentation import random_walker

# TODO: Add docstring
# TODO: Replace 'assert' statements with tests that raise ValueErrors
# TODO: Make sure indexing is correct and understandable

def get_patch_image(patch, img_shape, ctr):
    
    rx = (patch.shape[0] - 1) // 2
    ry = (patch.shape[1] - 1) // 2
    rz = (patch.shape[2] - 1) // 2
    
    cx = ctr[0]
    cy = ctr[1]
    cz = ctr[2]
    
    full_expand = np.zeros((img_shape[0] + 2 * rx, img_shape[1] + 2 * ry, img_shape[2] + 2 * rz))
    sl_x = slice(cx, cx + 2 * rx + 1)
    sl_y = slice(cy, cy + 2 * ry + 1)
    sl_z = slice(cz, cz + 2 * rz + 1)
    full_expand[sl_x, sl_y, sl_z] = patch
    
    return full_expand[rx:-rx, ry:-ry, rz:-rz]
    

def greedy_mp(img, flt, n_iter):
    
    # Check that filter dimensions are odd
    assert(flt.shape[0] % 2 == 1)
    assert(flt.shape[1] % 2 == 1)
    assert(flt.shape[2] % 2 == 1)
    
    # Check that L2 norm of filter is one
    assert(np.sum(flt ** 2) == 1)
    
    # Radii of original filter
    rx = (flt.shape[0] - 1) // 2
    ry = (flt.shape[1] - 1) // 2
    rz = (flt.shape[2] - 1) // 2
    
    # Convolve filter with itself
    flt_expand = np.zeros((4 * rx + 1, 4 * ry + 1, 4 * rz + 1))
    flt_expand[rx:-rx, ry:-ry, rz:-rz] = flt
    flt_conv = ndimage.filters.convolve(flt_expand, flt)
    
    # Convolve filter with image
    img_conv = ndimage.filters.convolve(img, flt)
    
    points = np.zeros((n_iter, 3), dtype=np.int)
    weights = np.zeros((n_iter, 1))

    for i in range(n_iter):
        
        # Get coordinates of pixel with maximum value in convolved image
        max_pt = np.unravel_index(np.argmax(img_conv), img_conv.shape)
        pix_val = img_conv[max_pt]
        
        # Subtract convolved filter from convolved image at chosen point
        img_conv -= get_patch_image(pix_val * flt_conv, img.shape, max_pt)
       
        # Add max point and corresponding weight to array
        points[i, :] = np.array(max_pt)
        weights[i] = pix_val
        
    return points, weights, img_conv


def get_gaussian_filter(cov, rads):

    # TODO: Add 'steps' parameters that allow us to control grid units

    # Radii of filter for x, y and z dimensions
    rx, ry, rz = rads
    
    # Grid for evaluating Gaussian on
    xg, yg, zg = np.mgrid[-rx:rx+1, -ry:ry+1, -rz:rz+1]
    flt_grid = np.stack((xg, yg, zg), axis=-1)

    # Gaussian filter (normalized to have unit L2 norm)
    flt_nn = multivariate_normal.pdf(flt_grid, mean=np.array([0, 0, 0]), cov=cov)
    flt = flt_nn / np.sqrt(np.sum(flt_nn ** 2))
    
    return flt


def mp_gaussian(img, cov, n_iter):
    
    # Create Gaussian filter with given covariance
    fl = get_gaussian_filter(cov, (15, 15, 5))

    # Run greedy MP algorithm on image using filter
    pts, weights, img_conv = greedy_mp(img, fl, n_iter)
    
    debug = {
        'fl': fl,
        'img_conv': img_conv
    }
    
    return pts, weights, debug


def watershed_gaussian(img):
    
    # Compute threshold with Otsu's method
    threshold_abs = threshold_otsu(img)
    idx_below_th = img < threshold_abs
    img_th = np.copy(img)
    img_th[idx_below_th] = 0.0
    
    # Find local peaks
    peaks = peak_local_max(img_th, min_distance=2)
    n_peaks = peaks.shape[0]
    
    # Run random walker segmentation algorithm on image, using peaks as starting points
    markers = np.zeros_like(img, dtype=np.int)
    markers[idx_below_th] = -1
    mark_vals = np.arange(n_peaks) + 1
    markers[peaks[:, 0], peaks[:, 1], peaks[:, 2]] = mark_vals
    img_seg = random_walker(img, markers)
    
    cell_indices = []
    for c in range(n_peaks):
        c_label = mark_vals[c]
        if np.count_nonzero(np.equal(img_seg, c_label)) >= CELL_MIN_SIZE:
            cell_indices.append(c)
    n_cells = len(cell_indices)
    
    # Create grid
    xg, yg, zg = np.mgrid[0:img.shape[0], 0:img.shape[1], 0:img.shape[2]]
    grid = np.stack((xg, yg, zg), axis=-1)

    # Compute mean and covariance for each cell
    cell_means = []
    cell_covs = []
    for cell in range(n_cells):
    
        # Index of cell
        c_idx = cell_indices[cell]
    
        # Boolean array representing 'footprint' of cell
        c_bin = np.equal(img_seg, mark_vals[c_idx])
    
        # Get position values and pixel values  
        pos_vals = grid[c_bin]
        pix_vals = img[c_bin]
            
        # Compute mean and covariance (adding ridge to covariance for conditioning)
        mean = np.average(pos_vals, axis=0, weights=pix_vals)
        cov = np.cov(pos_vals, aweights=pix_vals, rowvar=False) + np.eye(3) * 1e-5
        cell_means.append(mean)
        cell_covs.append(cov)
        
    # Evaluate Gaussian for each cell on grid to get discrete basis function
    basis_imgs = []
    for cell in range(n_cells):
    
        mean = cell_means[cell]
        cov = cell_covs[cell]
    
        # TODO: Either remove this from code or add it to other method so they can be accurately compared
        # Basis function is truncated Gaussian with double covariance
        rv = multivariate_normal(mean, cov * 2)
        basis = rv.pdf(grid)
        basis[basis < 1e-4] = 0
    
        basis_imgs.append(basis)
        
    # Get coefficients by solving least-squares system
    basis_vecs = np.hstack([x.reshape((-1, 1)) for x in basis_imgs])
    img_vec = img.reshape((-1, 1))
    coeff, r_sum, _, _  = np.linalg.lstsq(basis_vecs, img_vec, rcond=None)

    # TODO: Change this to return parameters of truncated Gaussians instead of whole images
    return coeff, basis_imgs