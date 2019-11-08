"""Functions for creating low-dimensional image representations"""

import numpy as np
from scipy import ndimage
from scipy.stats import multivariate_normal
from skimage.feature import peak_local_max
from skimage.util import img_as_float, img_as_uint
from skimage.measure import label
from skimage.filters import threshold_otsu
from skimage.segmentation import random_walker

# TODO: Add docstrings
# TODO: Replace 'assert' statements with tests that raise ValueErrors


def get_patch_image(patch, img_shape, ctr):
    """Create 3D image that is zero everywhere except for region around point
    
    If center point is close to edge of image, parts of patch that fall outside
    image boundaries will be discarded.
    
    Args:
        patch (numpy.ndarray): 3D Image patch to place at center. All
            dimensions must be odd.
        img_shape (tuple): Shape of image to place patch in
        ctr (numpy.ndarray): Coordinates in new image to place center of patch 
        
    Returns:
        numpy.ndarray: Image with patch inserted
    
    """
    
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


# TODO: Add 'steps' parameters that allow us to control grid units
def get_gaussian_filter(cov, rads):
    """Create array representing Gaussian filter with unit Euclidean norm
    
    Args:
        cov (numpy.ndarray): Covariance matrix
        rads (tuple): Radii of filter. The dimensions of the filter array will 
            be twice the radii plus one.
            
    Returns:
        numpy.ndarrray: Array containing filter values
    
    """

    # Radii of filter for x, y and z dimensions
    rx, ry, rz = rads
    
    # Grid for evaluating Gaussian on
    xg, yg, zg = np.mgrid[-rx:rx+1, -ry:ry+1, -rz:rz+1]
    flt_grid = np.stack((xg, yg, zg), axis=-1)

    # Gaussian filter (normalized to have unit L2 norm)
    flt_nn = multivariate_normal.pdf(flt_grid, mean=np.array([0, 0, 0]), cov=cov)
    flt = flt_nn / np.sqrt(np.sum(flt_nn ** 2))
    
    return flt
   
    
# TODO: Try to speed this up by 'expanding' img_recon to avoid re-allocating
# massive arrays full of zeros
def reconstruct_image(means, covs, weights, shape):
    """Reconstruct 3D image from weighted combination of Gaussian components
    
    Args:
        means (list of numpy.ndarrays): Means of components
        covs (list of numpy.ndarrays): Covariances of components
        weights (list of floats): Weights for components
        shape (tuple): Dimensions of image
    
    Returns:
        numpy.ndarray: Image representing weighted combination of components
    """
    
    img_recon = np.zeros(shape)
    for k in range(len(means)):
    
        cell = weights[k] * get_gaussian_filter(covs[k], (15, 15, 5))
        img_recon += get_patch_image(cell, shape, means[k])
    
    return img_recon


def greedy_mp(img, flt, n_iter):
    """Run greedy matching-pursuit algorithm (Elad, 2014) on image
    
    Args:
        img (numpy.ndarray): Image to extract components from
        flt (numpy.ndarray): Filter for algorithm. Euclidean norm of filter
            must be equal to one, and dimensions must be odd.
        n_iter (int): Number of iterations to run algorithm for
        
    Returns:
        list of numpy.ndarrays: Locations of components
        list of floats: Weights for components
        numpy.ndarray: Convolved residual (used for debugging)
    """
    
    # Check that filter dimensions are odd
    assert(flt.shape[0] % 2 == 1)
    assert(flt.shape[1] % 2 == 1)
    assert(flt.shape[2] % 2 == 1)
    
    # Check that L2 norm of filter is one
    assert(np.isclose(np.sum(flt ** 2), 1.0, atol=1e-5))
    
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
        
    # Not sure if we need this; just putting it in for now
    for w in weights:
        assert(w > 0)
        
    return points, weights, img_conv


def mp_gaussian(img, cov, n_iter):
    """Run greedy MP algorithm (Elad, 2014) with Gaussian filter on image
    
    Args:
        img (numpy.ndarray): Image to extract components from
        cov (numpy.ndarray): Covariance of Gaussian filter
        n_iter (int): Number of iterations
        
    Returns:
        list of numpy.ndarrays: Locations of components
        list of floats: Weights for components
        dict: Debug information:
            'fl' (numpy.ndarray): Gaussian filter used for algorithm
            'img_conv' (numpy.ndarray): Convolved residual
    """
    
    # Create Gaussian filter with given covariance
    fl = get_gaussian_filter(cov, (15, 15, 5))

    # Run greedy MP algorithm on image using filter
    pts, weights, img_conv = greedy_mp(img, fl, n_iter)
    
    debug = {
        'fl': fl,
        'img_conv': img_conv
    }
    
    return pts, weights, debug


# TODO: Clean up code; bring interface more in line with mp_gaussian() function
def watershed_gaussian(img):
    """Run watershed-based method for extracting Gaussian components from image
   
    Args:
        img (numpy.ndarray): Image to extract components from
    
    Returns:
        numpy.ndarray: Weights for components 
        list of numpy.ndarrays: Component images
    """
    
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