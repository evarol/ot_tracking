"""Classes for reading data from different formats"""

from abc import abstractmethod
from contextlib import AbstractContextManager

import h5py
import numpy as np
from skimage.util import img_as_float, img_as_ubyte


class WormDataReader(AbstractContextManager):
    """Abstract base class for classes that read worm data.
    
    This class inherits from AbstractContextManager, which 
    
    """
    
    @abstractmethod
    def get_frame(self, time):
        """Get frame from dataset corresponding to time point.
        
        Args:
            idx (int): Index of frame
            
        Returns:
            numpy.ndarray containing data for a single frame of video, with
                shape (X, Y, Z)
        
        """
        
        pass

    
class SyntheticReader(WormDataReader):
    """Reader for synthetic data"""
    
    def __init__(self, fpath):
        
        self.file = h5py.File(fpath, 'r')
        self.dset = self.file.get('video')
    
    def __exit__(self, exc_type, exc_value, traceback):
        
        self.file.close()
   
    def get_frame(self, time):

        return img_as_float(self.dset[:, :, :, time])


class ZimmerReader(WormDataReader):
    """Reader for data from Zimmer lab"""
    
    def __init__(self, fpath):
        
        self.file = h5py.File(fpath, 'r')
        self.dset = self.file.get('mCherry')
    
    def __exit__(self, exc_type, exc_value, traceback):
        
        self.file.close()
   
    def get_frame(self, time):

        frame_raw = self.dset[time, 0, :, :, :]
        frame_flip = np.moveaxis(frame_raw, [0, 1, 2], [2, 1, 0])

        return img_as_float(frame_flip)
    
    
class VivekReader(WormDataReader):
    """Reader for Vivek's data"""
    
    def __init__(self, fpath):
        
        self.file = h5py.File(fpath, 'r')
        self.dset = self.file.get('data')
    
    def __exit__(self, exc_type, exc_value, traceback):
        
        self.file.close()
   
    def get_frame(self, time):

        frame_raw = self.dset[time, :, :, :]
        frame_flip = np.moveaxis(frame_raw, [0, 1, 2], [2, 1, 0])

        return img_as_float(frame_flip)