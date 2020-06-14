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
    
    @property
    @abstractmethod
    def units(self):
        """Dimensions of voxel element, in microns"""
        
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


# TODO: Rethink format used to store synthetic data
# TODO: Add get_mp_frame() method to load MP data
class SyntheticReader(WormDataReader):
    """Reader for synthetic data"""
    
    def __init__(self, fpath):
        
        self._file = h5py.File(fpath, 'r')
        self._dset = self._file.get('video')
        self._num_frames = self._dset.shape[3]
        
        self._means = self._file.get('means')[:, :]
        self._cov = self._file.get('cov')[:, :]
        self._weights = self._file.get('weights')[:, :]
        
        # TODO: Eventually load units from file
        self._units = np.array([1.0, 1.0, 1.0])
    
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
    
    @property
    def units(self):
        return self._units
    
    @property
    def means(self):
        return self._means
    
    @property
    def cov(self):
        return self._cov
    
    @property
    def weights(self):
        return self._weights

    def get_frame(self, time):
        
        if time not in range(self.t_start, self.t_stop):
            raise ValueError('Invalid time value')

        return img_as_float(self._dset[:, :, :, time])
    

class ZimmerReader(WormDataReader):
    """Reader for data from Zimmer lab"""
    
    def __init__(self, fpath):
        
        self._file = h5py.File(fpath, 'r')
        self._dset = self._file.get('mCherry')
        self._num_frames = self._dset.shape[0]
        
        # Units in Zimmer files are not correct. Once this is fixed, we should
        # load units from file instead of using hard-coded value
        self._units = np.array([0.325, 0.325, 1.0])
    
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
    
    @property
    def units(self):
        return self._units
    
    def get_frame(self, time):
        
        if time not in range(self.t_start, self.t_stop):
            raise ValueError('Invalid time value')

        frame_raw = self._dset[time, 0, :, :, :]
        frame_flip = np.moveaxis(frame_raw, [0, 1, 2], [2, 1, 0])

        return img_as_float(frame_flip)


class VivekReader(WormDataReader):
    """Reader for Vivek's data"""
    
    def __init__(self, fpath):
        
        file_vars = loadmat(fpath, variable_names=['data', 'id_data'])
        self._data = file_vars['data']
        self._id_data = file_vars['id_data']
        
        self._num_frames = self._data.shape[3]
        self._units = self._id_data['info'][0][0][0][0][1].flatten()
    
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
    
    @property
    def units(self):
        return self._units
    
    def get_frame(self, time):
        
        if time not in range(self.t_start, self.t_stop):
            raise ValueError('Invalid time value')

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
        
        # TODO: Replace this with real numbers once I get them
        self._units = np.array([1.0, 1.0, 1.0])
        
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
    
    @property
    def units(self):
        return self._units
    
    def get_frame(self, time):
        
        if time not in range(self.t_start, self.t_stop):
            raise ValueError('Invalid time value')

        fpath = self._fpaths[time - self.t_start]
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


class MPWriter(AbstractContextManager):
    """Writer for matching pursuit (MP) representations of worm data."""
    
    def __init__(self, fpath):
        self._file = h5py.File(fpath, 'w')
        
    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
    
    def write(self, mps, t_start, t_stop):
        """Write MP frames to file.
        
        Args:
            mps (list of ImageMP objects): MP representations of video frames
            t_start (int): Time index of first frame
            t_stop (int): Time index one greater than last frame
            
        """
        
        # Assume covariance and limits are same for all MPs
        cov = mps[0].cov
        img_limits = mps[0].img_limits
        
        pts = np.array([x.pts for x in mps])
        wts = np.array([x.wts for x in mps])
        
        self._file.create_dataset('pts', data=pts)
        self._file.create_dataset('wts', data=wts)
        
        self._file.attrs['cov'] = cov
        self._file.attrs['img_limits'] = img_limits
        self._file.attrs['t_start'] = t_start
        self._file.attrs['t_stop'] = t_stop


class MPReader(AbstractContextManager):
    """Reader for matching pursuit (MP) representations of worm data."""
    
    def __init__(self, fpath):
        
        self._file = h5py.File(fpath, 'r')
        self._pts = self._file.get('pts')
        self._wts = self._file.get('wts')
        
        self._cov = self._file.attrs['cov']
        self._img_limits = self._file.attrs['img_limits']
        self._t_start = self._file.attrs['t_start']
        self._t_stop = self._file.attrs['t_stop']
        
    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()

    def get_frame(self, t):
        """Load MP representation of frame.
        
        Args:
            t (int): Time step to load frame for
        
        Returns:
            imagerep.ImageMP: MP representation for frame at time t
            
        Raises: 
            ValueError: If file doesn't contain frame for time t
        
        """

        if t not in range(self.t_start, self.t_stop):
            raise ValueError('Invalid time value')
            
        idx = t - self.t_start
        return imagerep.ImageMP(
            self._pts[idx, :, :],
            self._wts[idx, :],
            self._cov,
            self._img_limits
        )

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop