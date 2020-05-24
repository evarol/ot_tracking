"""Functions for creating low-dimensional image representations"""

import functools
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from scipy.stats import multivariate_normal
from skimage.feature import peak_local_max
from skimage.util import img_as_float, img_as_uint
from skimage.measure import label
from skimage.filters import threshold_otsu
from skimage.segmentation import random_walker

from otimage import io

# TODO: Add docstrings
# TODO: Replace 'assert' statements with tests that raise ValueErrors


# TODO: Add docstring
class ImageMP:
    
    def __init__(self, pts, wts, cov, img_shape):
        
        self._pts = pts
        self._wts = wts
        self._cov = cov
        self._img_shape = img_shape
        
    @property
    def pts(self):
        return self._pts

    @property
    def wts(self):
        return self._wts
    
    @property
    def cov(self):
        return self._cov
    
    @property
    def img_shape(self):
        return self._img_shape


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
        assert(w >= 0)
        
    return points, weights, img_conv


def mp_gaussian(img, cov, n_iter):
    """Run greedy MP algorithm (Elad, 2014) with Gaussian filter on image
    
    Args:
        img (numpy.ndarray): Image to extract components from
        cov (numpy.ndarray): Covariance of Gaussian filter
        n_iter (int): Number of iterations
        
    Returns:
        ImageMP: MP representation of image
        dict: Debug information:
            'fl' (numpy.ndarray): Gaussian filter used for algorithm
            'img_conv' (numpy.ndarray): Convolved residual
    """
    
    # Create Gaussian filter with given covariance
    fl = get_gaussian_filter(cov, (15, 15, 5))

    # Run greedy MP algorithm on image using filter
    pts, wts, img_conv = greedy_mp(img, fl, n_iter)
    
    mp = ImageMP(pts, wts, cov, img.shape)
    debug = {'fl': fl, 'img_conv': img_conv}
    
    return mp, debug


# TODO: Try to speed this up by 'expanding' img_recon to avoid re-allocating
# massive arrays full of zeros
def reconstruct_gaussian_image(pts, wts, cov, shape):
    """Reconstruct 3D image from weighted combination of Gaussian components"""
    
    # Only plot points that fall inside image
    plot_idx = np.all((pts >= 0) & (pts < shape), axis=1)
    pts_plot = (pts[plot_idx]).astype('int')
    wts_plot = wts[plot_idx]
    
    img_recon = np.zeros(shape)
    for k in range(pts_plot.shape[0]):
    
        cell = wts_plot[k] * get_gaussian_filter(cov, (15, 15, 5))
        img_recon += get_patch_image(cell, shape, pts_plot[k, :])
    
    return img_recon


def reconstruct_mp_image(mp):
    """Reconstruct 3D image from weighted combination of Gaussian components.
    
    Args:
        mp (ImageMP): MP representation of image
        
    Returns:
        np.ndarray: Reconstructed image
    """
    
    return reconstruct_gaussian_image(mp.pts, mp.wts, mp.cov, mp.img_shape)