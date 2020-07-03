"""Utility functions for notebooks"""

import numpy as np
import ot
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def pixel_dist_2d(nx, ny):
    """Compute pixel distance matrix for 2D image"""
    
    [xg, yg] = np.meshgrid(np.arange(nx), np.arange(ny))
    grid_vals = np.hstack((xg.reshape(-1, 1), yg.reshape(-1, 1)))
    
    return ot.dist(grid_vals, metric='sqeuclidean')


def pixel_dist_3d(nx, ny, nz):
    """Compute pixel distance matrix for 3D image"""
    
    [xg, yg, zg] = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))
    grid_vals = np.hstack(
        (xg.reshape(-1, 1), yg.reshape(-1, 1), zg.reshape(-1, 1)))
    
    return ot.dist(grid_vals, metric='sqeuclidean')


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
    
    # Compute pixel distance matrix 
    M = pixel_dist_2d(nx, ny)

    # Normalize distance matrix -- this prevents conditioning issues
    M = M / np.median(M)

    # Compute barycenter between images to use as interpolant
    A = np.hstack((img_1.reshape(-1, 1), img_2.reshape(-1, 1)))
    weights = np.array([1 - alpha, alpha])
    interp_vec = ot.bregman.barycenter(A, M, reg, weights)
    
    return interp_vec.reshape((nx, ny))


def make_movie(frames, fig):
    """Create animation from list of frames.
    
    Args:
        frames (numpy.ndarray iterable): 2D frames for video
        fig (matplotlib.Figure): Figure to use for animation
        
    Returns:
        animation.ArtistAnimation: Object containing video
    
    """
    
    plt.figure(fig.number)
    ims = [[plt.imshow(f, animated=True)] for f in frames]
    
    return animation.ArtistAnimation(
        fig, ims, interval=150, blit=True, repeat_delay=1000)


def plot_maxproj(img, ax=None, animated=False):
    """Plot max-projection of 3D image.
    
    Args:
        img (numpy.ndarray): Image to plot
        ax (matplotlib.axes.Axes): Optional. Axes to plot on (default is to
            use current axes).
        animated (bool): Optional. True if plot is meant to be part of
            animation, otherwise False.
    """
    
    if ax is None:
        ax = plt.axes()
    
    return ax.imshow(np.max(img, 2).T, origin='lower', animated=animated)

    
def plot_img_units(img, units, ax=None, animated=False):
    """Plot 2D image with given units.
    
    Args:
        img (numpy.ndarray): Image to plot
        units (numpy.ndarray): Units of grid (microns)
        ax (matplotlib.axes.Axes): Optional. Axes to plot on (default is to
            use current axes).
        animated (bool): Optional. True if plot is meant to be part of
            animation, otherwise False.
    """
     
    if ax is None:
        ax = plt.axes()
    
    xmax = img.shape[0] * units[0]
    ymax = img.shape[1] * units[1]
    extent = (0, xmax, 0, ymax)

    return ax.imshow(img.T, origin='lower', extent=extent, animated=animated)


def plot_maxproj_units(img, units, ax=None, animated=False):
    """Plot max-projection of 3D image.
    
    Args:
        img (numpy.ndarray): Image to plot
        units (numpy.ndarray): Units of grid (microns)
        ax (matplotlib.axes.Axes): Optional. Axes to plot on (default is to
            use current axes).
        animated (bool): Optional. True if plot is meant to be part of
            animation, otherwise False.
    """
    
    maxproj = np.max(img, 2)
    
    return plot_img_units(maxproj, units, ax=ax, animated=animated)


def plot_maxproj_zoom(img, units, extent, ax=None, animated=False):
    """Plot max-projection of 3D image.
    
    Args:
        img (numpy.ndarray): Image to plot
        units (numpy.ndarray): Units of grid (microns)
        extent (4-tuple): Region to plot: (xmin, xmax, ymin, ymax)
        ax (matplotlib.axes.Axes): Optional. Axes to plot on (default is to
            use current axes).
        animated (bool): Optional. True if plot is meant to be part of
            animation, otherwise False.
    """
        
    if ax is None:
        ax = plt.axes()
        
    xmin_vx = int(extent[0] / units[0])
    xmax_vx = int(extent[1] / units[0])
    ymin_vx = int(extent[2] / units[1])
    ymax_vx = int(extent[3] / units[1])
    
    img_zoom = img[xmin_vx:xmax_vx, ymin_vx:ymax_vx, :]
    maxproj = np.max(img_zoom, 2)

    return ax.imshow(maxproj.T, origin='lower', extent=extent, animated=animated)