"""Script for computing linear transforms from all frames to frame 1"""

import numpy as np
from scipy.io import loadmat, savemat

from otimage.imagereg import ot_reg_linear_1


# Zimmer
#MP_FPATH = '/home/mn2822/Desktop/WormOT/data/zimmer/mp_components/mp_0000_0050.mat'
#OUT_FPATH = '/home/mn2822/Desktop/WormOT/data/zimmer/registration/reg2_0003_0008.mat'

# Vivek
MP_FPATH = '/home/mn2822/Desktop/WormOT/data/vivek/0930_tail_01/mp_components/mp_0000_0900.mat'
OUT_FPATH = '/home/mn2822/Desktop/WormOT/data/vivek/0930_tail_01/registration/reg3_0000_0900.mat'

N_FRAMES = 899
N_MPS = 400


def main():

    print(f'Loading MP data from {MP_FPATH}...')

    mp_data = loadmat(MP_FPATH)
    pts = mp_data['means'][1:N_FRAMES+1, :, :]
    wts = mp_data['weights'][1:N_FRAMES+1, :, :]
    cov = mp_data['cov']

    print('Computing transformations...')

    # Points and weights for first frame
    pts_0 = pts[1, 0:N_MPS, :]
    wts_0 = wts[1, 0:N_MPS, 0]
    
    wts_unif = np.ones(N_MPS) / N_MPS
   
    f_alpha = [None] * N_FRAMES
    f_beta = [None] * N_FRAMES
    p_mtx = [None] * N_FRAMES

    f_alpha[0] = np.array([0, 0, 0])
    f_beta[0] = np.eye(3)
    p_mtx[0] = np.eye(N_MPS)

    for t in range(1, N_FRAMES):

        print(f'frame: {t}')

        pts_t = pts[t, 0:N_MPS, :]
        wts_t = wts[t, 0:N_MPS, 0]

        alpha, beta, log = ot_reg_linear_1(pts_0, pts_t, wts_unif, wts_unif)
        
        f_alpha[t] = alpha
        f_beta[t] = beta
        p_mtx[t] = log['P']

    print(f'Saving results to {OUT_FPATH}')
    mat_dict = {
        'fname': MP_FPATH,
        'n_frames': N_FRAMES,
        'n_mps': N_MPS,
        'f_alpha': np.array(f_alpha),
        'f_beta': np.array(f_beta),
        'p_mtx': np.array(p_mtx),
    }
    savemat(OUT_FPATH, mat_dict)


if __name__ == '__main__':
    main()
