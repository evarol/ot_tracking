"""Extract MP components from Vivek's data and write to MAT file"""

import numpy as np
from scipy.io import savemat

from readers import VivekReader
from imagerep import mp_gaussian


# Input and output paths
IN_FPATH = '/home/mn2822/Desktop/WormOT/data/vivek_old/animal_056_head/run401.mat'
OUT_FPATH = '/home/mn2822/Desktop/WormOT/data/vivek_old/animal_056_head/gmm_components/gmm_700_750.mat'

# Start and stop times for extraction
T_START = 700
T_STOP = 750

# Covariance values for each dimension
COV_DIAG = [4.0, 4.0, 1.0]

# Number of MP iterations to run
N_ITER = 500


def main():

    cov = np.diag(COV_DIAG)
    means = []
    weights = []

    with VivekReader(IN_FPATH) as reader:

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
