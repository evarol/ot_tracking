"""Extract MP components from Zimmer data and write to MAT file"""

import h5py
import numpy as np
from skimage.util import img_as_float
from scipy.io import savemat

from imagerep import mp_gaussian


# Input and output paths
IN_FPATH = '/home/mn2822/Desktop/WormTracking/data/zimmer/mCherry_v00065-01581.hdf5'
OUT_FPATH = '/home/mn2822/Desktop/WormTracking/data/zimmer/mat/test_vid.mat'

# Start and stop times for extraction
T_START = 500
T_STOP = 502

# Covariance values for each dimension
COV_DIAG = [4.0, 4.0, 1.0]

# Number of MP iterations to run
N_ITER = 10


def main():

    # Extract MP components from all frames
    cov = np.diag(COV_DIAG)
    means = []
    weights = []

    with h5py.File(IN_FPATH, 'r') as f:

        dset = f.get('mCherry')

        for t in range(T_START, T_STOP):

            print(f'Frame: {t}')

            img_raw = dset[t, 0, :, :, :]
            img_raw = np.moveaxis(img_raw, [0, 1, 2], [2, 1, 0])
            img = img_as_float(img_raw)

            mus, wts, _ = mp_gaussian(img, cov, N_ITER)
            means.append(mus)
            weights.append(wts)

    # Write to 
    mat_dict = {
        'means': np.array(means),
        'weights': np.array(weights),
        'cov': cov
    }
    savemat(OUT_FPATH, mat_dict)
    
if __name__ == '__main__':
    main()