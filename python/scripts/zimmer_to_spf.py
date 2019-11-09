"""Convert segment of Zimmer data to format ingestible by SPF"""

import numpy as np
import tifffile
import h5py
from skimage.util import img_as_float, img_as_ubyte

# Input and output paths
IN_FPATH = '/home/mn2822/Desktop/WormTracking/data/zimmer/mCherry_v00065-01581.hdf5'
OUT_DIR = '/home/mn2822/Desktop/WormTracking/data/zimmer/sample_video/tiff'

# Start and stop times for extraction
T_START = 500
T_STOP = 550


def load_frame(dset, t):
    """Load single frame from dataset"""

    img = dset[t, 0, :, :, :]
    img = np.moveaxis(img, [0, 1, 2], [2, 1, 0])

    return img_as_float(img)


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
    

def write_frame(k, img_ubyte):
    """Write frame to set of TIFF files"""

    for z in range(img_ubyte.shape[2]):

        fpath = f'{OUT_DIR}/image_t{k:04d}_z{z:04d}.tif'
        tifffile.imwrite(fpath, img_ubyte[:, :, z])

        
def main():
    
    with h5py.File(IN_FPATH, 'r') as f:

        print(f'Reading data from {IN_FPATH}...')
        dset = f.get('mCherry')

        vmin, vmax = get_video_range(dset, T_START, T_STOP)
        print(f'Min pixel value: {vmin}')
        print(f'Max pixel value: {vmax}')

        for k in range(T_STOP - T_START):

            t = T_START + k
            print(f'Frame: {t}')
            
            img = load_frame(dset, t)
            img_ubyte = convert_to_ubyte(img, vmin, vmax)
            write_frame(k, img_ubyte)
    

if __name__ == '__main__':
    main()
