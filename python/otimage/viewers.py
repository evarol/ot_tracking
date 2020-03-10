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

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    (Code originally copied from https://github.com/mohakpatel/ImageSliceViewer3D)
    
    """
    
    def __init__(self, volume, figsize=(8,8), cmap='plasma'):
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
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
            vmin=self.v[0], vmax=self.v[1])
        
        
class PushforwardViewer:
    """Viewer widget for transport plans"""
    
    def __init__(self, pts_1, pts_2, wts_1, wts_2, 
        cov, img_shape, p_mtx, figsize=(10, 10)):
        
        self.pts_1 = pts_1
        self.pts_2 = pts_2
        self.wts_1 = wts_1
        self.wts_2 = wts_2
        self.figsize = figsize
        
        self.rec_1 = imagerep.reconstruct_image(pts_1, [cov], wts_1, img_shape)
        self.rec_2 = imagerep.reconstruct_image(pts_2, [cov], wts_2, img_shape)
        
        q_mtx = p_mtx / np.sum(p_mtx, 1)
        self.pf_means = q_mtx @ pts_2
        self.pf_modes = pts_2[np.argmax(q_mtx, 1)]
        self.p_mtx = p_mtx
        self.q_mtx = q_mtx
        
        ipyw.interact(
            self.plot_pushforward, 
            idx=ipyw.IntSlider(
                min=0, max=pts_1.shape[0], step=1, 
                continuous_update=False, description='MP:'
            )
        )
        
    def plot_pushforward(self, idx):
        
        pt_1 = self.pts_1[idx, :]
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