"""Script for extracting GMMS from video, using parallelization"""

import numpy as np
from multiprocessing import Pool
from scipy.io import savemat

from readers import SyntheticReader
from imagerep import mp_gaussian


# Input and output paths
IN_FPATH = '/home/mn2822/Desktop/WormOT/data/synthetic/fast_3d/gmm_data_3d.h5'
OUT_FPATH = '/home/mn2822/Desktop/WormOT/data/synthetic/fast_3d/syn_data_mp.mat'

# Start and stop times for extraction
T_START = 5
T_STOP = 15

# Number of processes to use
N_PROCS = 1

# Covariance values for each dimension
COV_DIAG = [5.0, 5.0, 5.0]

# Number of MP iterations to run
N_ITER = 10


def get_gmms_mp(fpath, t_start, t_stop, cov, n_iter):

    means = []
    weights = []

    with SyntheticReader(fpath) as reader:
        
        for t in range(t_start, t_stop):

            # Load frame
            img = reader.get_frame(t)

            # Extract MP components from frame
            mus, wts, _ = mp_gaussian(img, cov, n_iter)
            means.append(mus)
            weights.append(wts)

    return (means, weights)


def get_chunks(t_start, t_stop, n_procs):

    # Determine size of chunks
    sz = (t_stop - t_start) // n_procs

    # First n-1 chunks
    chunks = []
    for k in range(n_procs - 1):
        chunks.append((t_start + k * sz, t_start + (k + 1) * sz))

    # Final chunk
    chunks.append((t_start + (n_procs - 1) * sz, t_stop))

    return chunks


def main():

    # Validate input
    if T_STOP <= T_START:
        raise ValueError('Start frame must be before stop frame')
    if N_PROCS <= 0:
        raise ValueError(f'{N_PROCS} is not a valid number of processes')
    if N_PROCS > (T_STOP - T_START):
        raise ValueError('More processes than frames')

    # Covariance matrix
    cov = np.diag(COV_DIAG)

    with Pool(processes=N_PROCS) as p:

        # Split frames into chunks for each process
        chunks = get_chunks(T_START, T_STOP, N_PROCS)

        # Run MP algorithm on frames across chunks
        results = p.map(
            lambda c: get_gmms_mp(IN_FPATH, c[0], c[1], cov, N_ITER),
            chunks
        )

        # Extract means and weights of components from result data
        means = [x[0] for r in results for x in r]
        weights = [x[1] for r in results for x in r]

    # Write means, weights, and covariance to MAT file
    mat_dict = {
        'means': np.array(means),
        'weights': np.array(weights),
        'cov': cov
    }
    savemat(OUT_FPATH, mat_dict)


if __name__ == '__main__':
    main()
