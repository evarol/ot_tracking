"""Classes for reading and writing data from"""

import os
import re
from abc import abstractmethod, ABC
from contextlib import AbstractContextManager
from operator import itemgetter

import h5py
import tifffile
import numpy as np
from scipy.io import loadmat, savemat
from skimage.util import img_as_float, img_as_ubyte

from otimage import imagerep


class WormDataReader(AbstractContextManager):
    """Abstract base class for classes that read worm data.
    
    This class inherits from AbstractContextManager, which allows
    WormDataReader subclasses to be used as context managers.
    
    """

    @property
    @abstractmethod
    def t_start(self):
        """Time value where video begins; inclusive (int)"""

        pass

    @property
    @abstractmethod
    def t_stop(self):
        """Time value where video ends; exclusive (int)"""

        pass

    @property
    @abstractmethod
    def num_frames(self):
        """Number of frames in video (int)"""

        pass
    
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

        self._file = h5py.File(fpath, 'r')
        self._dset = self._file.get('video')
        self._num_frames = self._dset.shape[3]
        
        self._means = self._file.get('means')[:, :]
        self._cov = self._file.get('cov')[:, :]
        self._weights = self._file.get('weights')[:, :]
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()

    @property
    def t_start(self):
        return 0

    @property
    def t_stop(self):
        return self._num_frames
    
    @property
    def means(self):
        return self._means
    
    @property
    def cov(self):
        return self._cov
    
    @property
    def weights(self):
        return self._weights

    @property
    def num_frames(self):
        return self._num_frames
   
    def get_frame(self, time):
        return img_as_float(self._dset[:, :, :, time])
    

class ZimmerReader(WormDataReader):
    """Reader for data from Zimmer lab"""
    
    def __init__(self, fpath):
        
        self._file = h5py.File(fpath, 'r')
        self._dset = self._file.get('mCherry')
        self._num_frames = self._dset.shape[0]
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()

    @property
    def t_start(self):
        return 0

    @property
    def t_stop(self):
        return self._num_frames

    @property
    def num_frames(self):
        return self._num_frames
   
    def get_frame(self, time):

        frame_raw = self._dset[time, 0, :, :, :]
        frame_flip = np.moveaxis(frame_raw, [0, 1, 2], [2, 1, 0])

        return img_as_float(frame_flip)


class VivekReader(WormDataReader):
    """Reader for Vivek's data"""
    
    def __init__(self, fpath):
        
        file_vars = loadmat(fpath, variable_names=['data'])
        self._data = file_vars['data']
        self._num_frames = self._data.shape[3]
    
    def __exit__(self, exc_type, exc_value, traceback):
        return None

    @property
    def t_start(self):
        return 0

    @property
    def t_stop(self):
        return self._num_frames

    @property
    def num_frames(self):
        return self._num_frames
   
    def get_frame(self, time):
        return img_as_float(self._data[:, :, :, time])


class HillmanReader(WormDataReader):
    """Reader for Hillman lab data"""
    
    def __init__(self, dirpath):

        # Get list of all TIFF files in directory
        pattern = re.compile(r'_t\d{5}\.tif')
        fpaths = [
            e.path for e in os.scandir(dirpath) 
            if e.is_file() and pattern.search(e.path)]
        
        if not fpaths:
            raise ValueError(f'Directory contains no frame files: {dirpath}')
        
        # Extract time values for each file and sort by time
        time_fpath = [(int(p[-9:-4]), p) for p in fpaths]
        time_fpath_sorted = sorted(time_fpath, key=itemgetter(0))
        times_sorted, fpaths_sorted = zip(*time_fpath_sorted)

        # Check that directory contains continuous time series
        t_start = times_sorted[0]
        t_stop = times_sorted[-1] + 1
        
        # Make sure no frames are missing
        missing_frames = set(range(t_start, t_stop)) - set(times_sorted) 
        if missing_frames:
            raise ValueError(f'Directory is missing frames: {missing_frames}')
    
        self._fpaths = fpaths_sorted
        self._t_start = t_start
        self._t_stop = t_stop
        self._num_frames = t_stop - t_start
        
    def __exit__(self, exc_type, exc_value, traceback):
        return None

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop
    
    @property
    def num_frames(self):
        return self._num_frames
   
    def get_frame(self, time):
        
        if time not in range(self._t_start, self._t_stop):
            raise ValueError('Invalid time value')

        fpath = self._fpaths[time - self._t_start]
        frame_raw = tifffile.imread(fpath)
        frame_flip = np.moveaxis(frame_raw, [0, 1, 2], [2, 1, 0])

        return img_as_float(frame_flip)


class WormDataReaderFactory(ABC):
    """Abstract base class for WormDataReader factory."""
    
    @abstractmethod
    def get_reader(self):
        """Return WormDataReader object for this object's filepath"""
        
        pass
 

class SyntheticReaderFactory(WormDataReaderFactory):
    """Create SyntheticReader objects for single filepath"""
    
    def __init__(self, fpath):
        self._fpath = fpath
        
    def get_reader(self):
        return SyntheticReader(self._fpath)

    
class ZimmerReaderFactory(WormDataReaderFactory):
    """Create ZimmerReader objects for single filepath"""
    
    def __init__(self, fpath):
        self._fpath = fpath
        
    def get_reader(self):
        return ZimmerReader(self._fpath)
    
    
class VivekReaderFactory(WormDataReaderFactory):
    """Create VivekReader objects for single filepath"""
    
    def __init__(self, fpath):
        self._fpath = fpath
        
    def get_reader(self):
        return VivekReader(self._fpath)
 
    
class HillmanReaderFactory(WormDataReaderFactory):
    """Create HillmanReader objects for single filepath"""
    
    def __init__(self, fpath):
        self._fpath = fpath
        
    def get_reader(self):
        return HillmanReader(self._fpath)
 

class MPReader(AbstractContextManager):
    """Reader for matching pursuit (MP) representations of worm data."""
    
    def __init__(self, fpath):
        
        data = loadmat(fpath)
        
        self._pts = data['means']
        self._wts = data['weights']
        self._cov = data['cov']
        self._t_start = data['t_start']
        self._t_stop = data['t_stop']
        self._img_shape = (
            data['img_shape'][0, 0],
            data['img_shape'][0, 1],
            data['img_shape'][0, 2]
        )

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def get_frame(self, t):
        return imagerep.ImageMP(
            self._pts[t, :, :],
            self._wts[t, :, 0],
            self._cov,
            self._img_shape
        )

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop
    
    
class MPWriter(AbstractContextManager):
    """Writer for matching pursuit (MP) representations of worm data."""
    
    def __init__(self, fpath):
        self._file = open(fpath, 'wb')
        
    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
    
    def write(self, mps, t_start, t_stop):
        
        cov = mps[0].cov
        img_shape = mps[0].img_shape
        pts = np.array([x.pts for x in mps])
        wts = np.array([x.wts for x in mps])
        
        mat_data = {
            't_start': t_start,
            't_stop': t_stop,
            'cov': cov,
            'img_shape': img_shape,
            'means': pts,
            'weights': wts,
        }
        savemat(self._file, mat_data)
    
    