#!/usr/bin/env python3
"""Script for downloading microscopy data from Hillman Lab server.

This script reads microscopy data stored in TIFF files on the Hillman Lab's
server (known as 'Dingus') and stores in an HDF5 file on the local machine.

"""

import argparse
import os
import getpass
import tempfile
import numpy as np
from datetime import datetime as dt
from contextlib import contextmanager

import paramiko
import tifffile
import h5py


# Hostname for Dingus server
DINGUS_HOSTNAME = 'dingus.hillman.zi.columbia.edu'

# Username for Dingus server
DINGUS_USERNAME = 'amin'

# Port to reach SFTP protocol on
DINGUS_SFTP_PORT = 22

# Directory on Dingus where TIFF data is stored
DINGUS_DATA_DIR = (
    '/local_mount/space/dingus/2/Richard/'
    'Richard_wrom_Venkpaper_060719/tiff_stacks/worm3_run4/'
    '19-06-09 190446_flip_duo_skewed-54_dsf1_180secs'
);


def parse_args():
    """Parse arguments from command line"""

    parser = argparse.ArgumentParser(
        description='Fetch worm data from Dingus server.')

    parser.add_argument(
        '-s', '--start',
        type=int,
        required=True,
        help='index of first video frame'
    )
    parser.add_argument(
        '-e', '--end',
        type=int,
        required=True,
        help='index of last video frame'
    )
    parser.add_argument(
        '-o', '--out',
        required=True,
        help='local filepath to write data to'
    )

    return parser.parse_args()

@contextmanager
def dingus_sftp(password):
    """Context manager for SFTP connection to Dingus server"""

    try:
        transport = paramiko.Transport((DINGUS_HOSTNAME, DINGUS_SFTP_PORT))
        transport.connect(None, DINGUS_USERNAME, password)
        yield paramiko.SFTPClient.from_transport(transport)

    finally:
        transport.close()

@contextmanager
def temp_path():
    """Context manager for path to temporary file"""

    fp, fpath = tempfile.mkstemp()

    yield fpath

    os.close(fp)
    os.remove(fpath)

def read_tiff_files(fnames, dirname, sftp):
    """Read data from TIFF files on server into NumPy array"""

    frames = []

    for fname in fnames:

        print(f'file: {fname}')

        # Weird hack: Paramiko and Tifffile libraries cause error if SFTP
        # file handle is passed as argument to tifffile.imread(). Work-around
        # is to create temp file with visible OS path, write file to it using
        # SFTP, and then read file from path using tiffile.imread(). This is
        # the only approach to this problem I've found that works.
        with temp_path() as fpath_tmp:
            sftp.get(f'{dirname}/{fname}', fpath_tmp)
            frames.append(tifffile.imread(fpath_tmp))

    # Convert from T*Z*X*Y array to X*Y*Z*T array
    return np.moveaxis(np.array(frames), [0, 1, 2, 3], [3, 2, 1, 0])

def main():

    args = parse_args()
    times = range(args.start, args.end + 1)
    out_fpath = args.out

    print(f'Accessing server at {DINGUS_HOSTNAME}...')
    print(f'username: {DINGUS_USERNAME}')
    password = getpass.getpass('password: ')

    with dingus_sftp(password) as sftp:

        print(f'Reading files from {DINGUS_DATA_DIR}...')

        r_fnames = [f'R_worm3_run4_t{t:012d}.tiff' for t in times]
        g_fnames = [f'G_worm3_run4_t{t:012d}.tiff' for t in times]

        r_data = read_tiff_files(r_fnames, DINGUS_DATA_DIR, sftp)
        g_data = read_tiff_files(g_fnames, DINGUS_DATA_DIR, sftp)

    print(f'Writing data to {out_fpath}...')

    with h5py.File(out_fpath, 'w') as f:

        f.create_dataset('red', data=r_data)
        f.create_dataset('green', data=g_data)

        f.attrs['t_start'] = args.start
        f.attrs['t_end'] = args.end
        f.attrs['data_dir'] = DINGUS_DATA_DIR
        f.attrs['time_retrieved_utc'] = \
            dt.utcnow().replace(microsecond=0).isoformat()

    print('Done')

if __name__ == '__main__':
    main()
