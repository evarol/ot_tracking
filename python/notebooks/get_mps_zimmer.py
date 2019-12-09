"""Extract MP components from Zimmer data and write to MAT file"""

import argparse

import h5py
import numpy as np
from skimage.util import img_as_float
from scipy.io import savemat

from readers import ZimmerReader
from imagerep import mp_gaussian


# Input and output paths
IN_FPATH = '/home/mn2822/Desktop/WormOT/data/zimmer/mCherry_v00065-01581.hdf5'
OUT_DIR = '/home/mn2822/Desktop/WormOT/data/zimmer/all_gmms'

# Covariance values for each dimension
COV_DIAG = [4.0, 4.0, 1.0]

# Number of MP iterations to run
N_ITER = 500


def parse_args():

    parser = argparse.ArgumentParser(description='Compute GMMs for Zimmer data')
    parser.add_argument('--start', '-s', type=int, help='frame to start at')
    parser.add_argument('--end', '-e', type=int, help='frame to end at')

    return parser.parse_args()


def get_gmms(fpath, t_start, t_stop, cov, n_iter):

    means = []
    weights = []

    with ZimmerReader(fpath) as rdr:

        for t in range(t_start, t_stop):

            print(f'Frame: {t}')

            # Load frame
            img = rdr.get_frame(t)

            # Extract MP components from frame
            mus, wts, _ = mp_gaussian(img, cov, n_iter)
            means.append(mus)
            weights.append(wts)

    return (means, weights)


def main():

    # Parse arguments
    args = parse_args()
    t_start = args.start
    t_stop = args.end

    # Compute GMMs
    cov = np.diag(COV_DIAG)
    means, weights = get_gmms(IN_FPATH, t_start, t_stop, cov, N_ITER)

    # Write means, weights, and covariance to MAT file
    out_fpath = f'{OUT_DIR}/vid_{t_start:04}_{t_stop:04}.mat'
    mat_dict = {
        'means': np.array(means),
        'weights': np.array(weights),
        'cov': cov
    }
    savemat(out_fpath, mat_dict)
    
    
if __name__ == '__main__':
    main()
