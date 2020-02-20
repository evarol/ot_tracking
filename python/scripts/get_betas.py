"""Script for computing frame-to-frame transforms"""

import numpy as np
from scipy.io import loadmat, savemat

from otimage import imagereg


MP_FPATH = '/home/mn2822/Desktop/WormOT/data/zimmer/gmm_components/vid_0500_0550.mat'
OUT_FPATH = '/home/mn2822/Desktop/WormOT/data/zimmer/mappings/map_0500_0550.mat'
N_FRAMES = 50
N_MPS = 100


def main():

    print(f'Loading MP data from {MP_FPATH}...')
    mp_data = loadmat(MP_FPATH)
    pts = mp_data['means']
    wts = mp_data['weights']
    cov = mp_data['cov']

    print('Computing transformations...')
    alphas = []
    betas = []

    for t in range(N_FRAMES - 1):

        print(f'frames: {t} -> {t+1}')

        pts_1 = pts[t, 0:N_MPS, :]
        wts_1 = wts[t, 0:N_MPS, 0]
        pts_2 = pts[t + 1, 0:N_MPS, :]
        wts_2 = wts[t + 1, 0:N_MPS, 0]

        alpha, beta, _ = imagereg.ot_reg_linear(pts_1, pts_2, wts_1, wts_2)

        alphas.append(alpha)
        betas.append(beta)

    print(f'Saving results to {OUT_FPATH}')
    mat_dict = {
        'fname': MP_FPATH,
        'n_frames': N_FRAMES,
        'n_mps': N_MPS,
        'alphas': np.array(alphas),
        'betas': np.array(betas),
    }
    savemat(OUT_FPATH, mat_dict)


if __name__ == '__main__':
    main()
