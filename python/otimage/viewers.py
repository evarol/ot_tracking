"""Utilities for viewing data in notebooks"""

import numpy as np
import ipywidgets as ipyw
import matplotlib.pyplot as plt

from otimage import imagerep


class ImageSliceViewer3D:
    """ 
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Arguments:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    (Code originally copied from https://github.com/mohakpatel/ImageSliceViewer3D)
    
    """
    
    def __init__(self, volume, figsize=(8,8), cmap='plasma', origin='upper'):
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        self.origin = origin
        self.v = [np.min(volume), np.max(volume)]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1], origin=self.origin)


class WormSliceViewer:
    
    def __init__(self, img, units, figsize=(8,8)):
        
        self.img = img
        self.figsize = figsize
        self.img_min = np.min(img)
        self.img_max = np.max(img)
        
        # Image transposed so that first two indices correspond to slicing plane
        self.img_view = None
        self.extent_view = None
        
        # Extent dimensions for all three viewing planes
        xmax = img.shape[0] * units[0]
        ymax = img.shape[1] * units[1]
        zmax = img.shape[2] * units[2]
        self.extent_xy = (0, xmax, 0, ymax)
        self.extent_xz = (0, xmax, 0, zmax)
        self.extent_yz = (0, ymax, 0, zmax)
        
        # Widget for selecting plane
        plane_widget = ipyw.RadioButtons(
            options=['XY','XZ', 'YZ'], 
            value='XY', 
            description='plane:', 
            disabled=False,
            style={'description_width': 'initial'}
        )
        ipyw.interact(self.view_selection, plane=plane_widget)
        
    def view_selection(self, plane):
        
        if plane == 'XY':
            self.img_view = self.img
            self.extent_view = self.extent_xy
        elif plane == 'XZ':
            self.img_view = np.transpose(self.img, [0, 2, 1])
            self.extent_view = self.extent_xz
        elif plane == 'YZ':
            self.img_view = np.transpose(self.img, [1, 2, 0])
            self.extent_view = self.extent_yz
        else:
            raise ValueError('Invalid plane')
        
        # Widget for selecting slice
        z_max = self.img_view.shape[2] - 1
        slice_widget = ipyw.IntSlider(
            min=0, max=z_max, step=1, 
            continuous_update=False, 
            description='slice:'
        )
        ipyw.interact(self.plot_slice, idx=slice_widget)
        
    def plot_slice(self, idx):
        
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(
            self.img_view[:, :, idx].T, 
            extent=self.extent_view,
            vmin=self.img_min, 
            vmax=self.img_max, 
            origin='lower'
        )
 

class CellLocationViewer:
    
    def __init__(self, img, pts, units, figsize=(8,8)):
        
        self.img = img
        self.pts = pts
        self.figsize = figsize
        
        self.img_min = np.min(img)
        self.img_max = np.max(img)
        
        xmax = img.shape[0] * units[0]
        ymax = img.shape[1] * units[1]
        self.extent_xy = (0, xmax, 0, ymax)
        
        # Widget for selecting slice
        z_max = self.img.shape[2] - 1
        slice_widget = ipyw.IntSlider(
            min=0, max=z_max, step=1, 
            continuous_update=False, 
            description='slice:'
        )
        ipyw.interact(self.plot_slice, idx=slice_widget)

    def get_pts_slice(self, idx):
        
        return self.pts[self.pts[:, 2] == idx]

    def plot_slice(self, idx):
        
        pts_slice = self.get_pts_slice(idx)
        
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(
            self.img[:, :, idx].T, 
            extent=self.extent_xy,
            vmin=self.img_min, 
            vmax=self.img_max, 
            origin='lower'
        )
        plt.scatter(pts_slice[:, 0], pts_slice[:, 1], color='red', marker='*')
    
        
class PushforwardViewer:
    """Viewer widget for transport plans"""
    
    def __init__(self, mp_1, mp_2, q_mtx, units, figsize=(10, 10)):
        
        self.figsize = figsize
        self.mp_1 = mp_1
        self.mp_2 = mp_2
        self.rec_1 = imagerep.reconstruct_mp_image(mp_1, units)
        self.rec_2 = imagerep.reconstruct_mp_image(mp_2, units)
        
        self.pts_1_vx = mp_1.pts / units
        self.pts_2_vx = mp_2.pts / units
        self.n_pts = mp_1.pts.shape[0]
        
        self.pf_means = q_mtx @ self.pts_2_vx
        self.pf_modes = self.pts_2_vx[np.argmax(q_mtx, 1)]
        self.q_mtx = q_mtx
        
        ipyw.interact(
            self.plot_pushforward, 
            idx=ipyw.IntSlider(
                min=0, max=self.n_pts, step=1, 
                continuous_update=False, description='MP:'
            )
        )
        
    def plot_pushforward(self, idx):
        """Plot pushforward of MP component with given index."""
        
        pt_1 = self.pts_1_vx[idx, :]
        mean_pf = self.pf_means[idx, :]
        mode_pf = self.pf_modes[idx, :]

        plt.figure(figsize=(15, 15))

        plt.subplot(121)
        plt.imshow(np.max(self.rec_1, 2).T, origin='lower')
        plt.plot(pt_1[0], pt_1[1], marker='*', color='red', markersize=7)
        plt.axis('off')
        plt.title(f'MP: {idx}')

        plt.subplot(122)
        plt.imshow(np.max(self.rec_2, 2).T, origin='lower')
        plt.plot(mean_pf[0], mean_pf[1], marker='*', color='red', markersize=7)
        plt.plot(mode_pf[0], mode_pf[1], marker='+', color='red', markersize=7)
        plt.axis('off')
        plt.title('pushforward')