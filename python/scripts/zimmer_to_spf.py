"""Convert segment of Zimmer data to format ingestible by SPF"""

import numpy as np
import tifffile
import h5py
from skimage.util import img_as_float, img_as_ubyte

# Input and output paths
IN_FPATH = '/home/mn2822/Desktop/WormTracking/data/zimmer/mCherry_v00065-01581.hdf5'
OUT_DIR = '/home/mn2822/Desktop/WormTracking/data/zimmer/tiff'

# Start and stop times for extraction
T_START = 500
T_STOP = 501


def convert_to_ubyte(img):
    
    img_float = img_as_float(img)
    
    # Scale to full skimage float range to minimize loss during compression
    img_max = np.max(img_float)
    img_min = np.min(img_float)
    img_scl = (img_float - img_min) / (img_max - img_min)
    
    return img_as_ubyte(img_scl)


def write_frame(img, t):
    
    for z in range(img.shape[2]):
        
        fpath = f'{OUT_DIR}/image_t{t:04d}_z{z:04d}.tif'
        tifffile.imwrite(fpath, img[:, :, z])

        
def main():
    
    
    with h5py.File(IN_FPATH, 'r') as f:

        dset = f.get('mCherry')

        ## TODO: Get min and max of image here
    
        for t in range(T_START, T_STOP):

            print(f'Frame: {t}')

            # Load frame
            img = dset[t, 0, :, :, :]
            img = np.moveaxis(img, [0, 1, 2], [2, 1, 0])
            
            # Convert image to 8-bit
            img_ubyte = convert_to_ubyte(img)
           
            # Write frame to TIFF files
            write_frame(img_ubyte, t)
            
    
if __name__ == '__main__':
    main()