"""Extract MP components from synthetic data and write to MAT file"""

import h5py
import numpy as np
from skimage.util import img_as_float
from scipy.io import savemat

from imagerep import mp_gaussian, reconstruct_image


# Input and output paths
IN_FPATH = '/home/mn2822/Desktop/WormTracking/data/synthetic/gmm_data_3d.h5'
OUT_FPATH = '/home/mn2822/Desktop/WormTracking/data/synthetic/syn_data_mp.mat'

# Start and stop times for extraction
T_START = 0
T_STOP = 50

# Covariance values for each dimension
#COV_DIAG = [4.0, 4.0, 1.0]
COV_DIAG = [5.0, 5.0, 5.0]

# Number of MP iterations to run
#N_ITER = 500
N_ITER = 20


def main():

    cov = np.diag(COV_DIAG)
    means = []
    weights = []

    with h5py.File(IN_FPATH, 'r') as f:

        dset = f.get('red')

        #for t in range(T_START, T_STOP):
        for t in range(1):

            print(f'Frame: {t}')

            # Load frame
            img_raw = dset[:, :, :, t]
            img = img_as_float(img_raw)

            # Extract MP components from frame
            mus, wts, _ = mp_gaussian(img, cov, N_ITER)
            means.append(mus)
            weights.append(wts)
            
    img_recon = reconstruct_image(mus, wts, [cov] * N_ITER, img.shape)
    
    plt.subplot(121)
    plt.imshow(np.max(img, 2).T)
    
    plt.subplot(122)
    plt.imshow(np.max(img_recon, 2).T)

    # Write means, weights, and covariance to MAT file
    #mat_dict = {
    #    'means': np.array(means),
    #    'weights': np.array(weights),
    #    'cov': cov
    #}
    #savemat(OUT_FPATH, mat_dict)
    
    
if __name__ == '__main__':
    main()
