"""Convert synthetic data to format ingestible by SPF"""

import numpy as np
import tifffile
import h5py
from skimage.util import img_as_float, img_as_ubyte

# Input and output paths
IN_FPATH = '/home/mn2822/Desktop/WormTracking/data/synthetic/gmm_data_3d.h5'
OUT_DIR = '/home/mn2822/Desktop/WormTracking/data/synthetic/tiff'


def load_frame(dset, t):
    """Load single frame from dataset"""

    return img_as_float(dset[:, :, :, t])


def get_video_range(dset, t_start, t_stop):
    """Compute min and max pixel values across whole video"""

    min_vals = []
    max_vals = []

    for t in range(t_start, t_stop):

        img = load_frame(dset, t)
        min_vals.append(np.min(img))
        max_vals.append(np.max(img))

    return (min(min_vals), max(max_vals))


def convert_to_ubyte(img, vmin, vmax):
    """Convert image from float to unsigned byte"""
            
    img_scl = (img - vmin) / (vmax - vmin)

    return img_as_ubyte(img_scl)
    

def write_frame(t, img_ubyte):
    """Write frame to set of TIFF files"""

    for z in range(img_ubyte.shape[2]):
        
        fpath = f'{OUT_DIR}/image_t{t+1:04d}_z{z+1:04d}.tif'
        tifffile.imwrite(fpath, img_ubyte[:, :, z])
        
        
def main():
    
    with h5py.File(IN_FPATH, 'r') as f:

        print(f'Reading data from {IN_FPATH}...')
        dset = f.get('video')
        n_frames = dset.shape[3]

        vmin, vmax = get_video_range(dset, 0, n_frames)
        print(f'Min pixel value: {vmin}')
        print(f'Max pixel value: {vmax}')

        for t in range(n_frames):

            print(f'Frame: {t}')
            
            img = load_frame(dset, t)
            img_ubyte = convert_to_ubyte(img, vmin, vmax)
            write_frame(t, img_ubyte)
    

if __name__ == '__main__':
    main()
