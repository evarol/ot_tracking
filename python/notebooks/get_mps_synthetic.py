"""Extract MP components from synthetic data and write to file"""

import numpy as np
from scipy.io import savemat

from readers import SyntheticReader
from imagerep import mp_gaussian


# Input and output paths
IN_FPATH = '/home/mn2822/Desktop/WormOT/data/synthetic/fast_3d/gmm_data_3d.h5'
OUT_FPATH = '/home/mn2822/Desktop/WormOT/data/synthetic/fast_3d/syn_data_mp.mat'

# Start and stop times for extraction
T_START = 0
T_STOP = 1

# Covariance values for each dimension
COV_DIAG = [5.0, 5.0, 5.0]

# Number of MP iterations to run
N_ITER = 10


def main():

    cov = np.diag(COV_DIAG)
    means = []
    weights = []

    with SyntheticReader(IN_FPATH) as reader:
        
        for t in range(T_START, T_STOP):

            print(f'Frame: {t}')

            # Load frame
            img = reader.get_frame(t)

            # Extract MP components from frame
            mus, wts, _ = mp_gaussian(img, cov, N_ITER)
            means.append(mus)
            weights.append(wts)
            
    # Write means, weights, and covariance to MAT file
    mat_dict = {
        'means': np.array(means),
        'weights': np.array(weights),
        'cov': cov
    }
    savemat(OUT_FPATH, mat_dict)
    
    
if __name__ == '__main__':
    main()