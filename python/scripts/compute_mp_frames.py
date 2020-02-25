"""Script for extracting MPs from frames in video for OT regression"""

import numpy as np
import h5py

from otimage import readers, imagerep

IN_FPATH = '/home/mn2822/Desktop/WormOT/data/zimmer/raw/mCherry_v00065-00115.hdf5'
OUT_FPATH = '/home/mn2822/Desktop/WormOT/data/zimmer/sample/frame_mp_0000_0002.h5'

def main():

    with readers.ZimmerReader(IN_FPATH) as reader:
        
        frame_1 = reader.get_frame(0)
        frame_2 = reader.get_frame(1)
        
    n_iter = 200
    cov = np.diag([8.0, 8.0, 1.5])
    
    pts_1, wts_1, dbg_1 = imagerep.mp_gaussian(frame_1, cov, n_iter)
    pts_2, wts_2, dbg_2 = imagerep.mp_gaussian(frame_2, cov, n_iter)
    
    with h5py.File(OUT_FPATH, 'w') as f:
        
        f.create_dataset('n_iter', data=n_iter)
        f.create_dataset('cov', data=cov)
        
        grp_1 = f.create_group('frame_1')
        grp_1.create_dataset('img', data=frame_1)
        grp_1.create_dataset('pts', data=np.array(pts_1))
        grp_1.create_dataset('wts', data=np.array(wts_1))
        grp_1.create_dataset('fl', data=dbg_1['fl'])
        grp_1.create_dataset('img_conv', data=dbg_1['img_conv'])
        
        grp_2 = f.create_group('frame_2')
        grp_2.create_dataset('img', data=frame_2)
        grp_2.create_dataset('pts', data=np.array(pts_2))
        grp_2.create_dataset('wts', data=np.array(wts_2))
        grp_2.create_dataset('fl', data=dbg_2['fl'])
        grp_2.create_dataset('img_conv', data=dbg_2['img_conv'])

if __name__ == '__main__':
    main()