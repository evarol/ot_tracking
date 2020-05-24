"""Script for computing linear transforms from all frames to frame 1

This script computes transforms by finding transforms between each pair of 
successive frames and then composing them.

"""

import numpy as np
from scipy.io import loadmat, savemat

from otimage.imagereg import ot_reg_linear_1


MP_FPATH = '/home/mn2822/Desktop/WormOT/data/zimmer/mp_components/mp_0000_0050.mat'
OUT_FPATH = '/home/mn2822/Desktop/WormOT/data/zimmer/registration/reg1_0000_0050.mat'
N_FRAMES = 50
N_MPS = 100


def main():

    print(f'Loading MP data from {MP_FPATH}...')

    mp_data = loadmat(MP_FPATH)
    pts = mp_data['means']
    wts = mp_data['weights']
    cov = mp_data['cov']

    print('Computing transformations...')

    # Points and weights for first frame
    pts_0 = pts[0, 0:N_MPS, :]
    wts_0 = wts[0, 0:N_MPS, 0]
    
    f_alpha = [None] * N_FRAMES
    f_beta = [None] * N_FRAMES
    rec_pts = [None] * N_FRAMES

    f_alpha[0] = np.array([0, 0, 0])
    f_beta[0] = np.eye(3)
    rec_pts[0] = pts_0

    for t in range(1, N_FRAMES):

        print(f'frame: {t}')
        
        # Points and weight distribution for previous frame
        pts_1 = pts[t - 1, 0:N_MPS, :]
        wts_1 = wts[t - 1, 0:N_MPS, 0]

        # Points and weight distribution for current frame
        pts_2 = pts[t, 0:N_MPS, :]
        wts_2 = wts[t, 0:N_MPS, 0]

        # Compute mapping from reconstruction of prev. frame to current frame
        alpha, beta, _ = ot_reg_linear_1(pts_1, pts_2, wts_1, wts_2)
        
        print('det(beta)')
        print(np.linalg.det(beta))
        
        # Compute mapping from first frame to current frame using recursive update
        f_alpha[t] = beta @ f_alpha[t - 1] + alpha
        f_beta[t] = beta @ f_beta[t - 1]

        # Use mapping to reconstruct current frame from first frame
        rec_pts[t] = f_alpha[t] + pts_0 @ f_beta[t].T

    print(f'Saving results to {OUT_FPATH}')
    mat_dict = {
        'fname': MP_FPATH,
        'n_frames': N_FRAMES,
        'n_mps': N_MPS,
        'f_alpha': np.array(f_alpha),
        'f_beta': np.array(f_beta),
        'rec_pts': np.array(rec_pts),
    }
    savemat(OUT_FPATH, mat_dict)


if __name__ == '__main__':
    main()
