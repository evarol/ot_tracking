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