"""Compute Gaussian MPs for Vivek data"""

import numpy as np

from otimage import io, mp_parallel


# Input file
in_fpath = '/home/mn2822/Desktop/WormTraces/data/Vivek/1024_tail_06/data.mat'

# Output file
out_fpath = '/home/mn2822/Desktop/WormOT/data/vivek/1024_tail_06/mp_components/mp_0000_0900.h5'

# Range of frames to compute MPs for
t_start = 0
t_stop = 900

# Number of processes to launch
n_procs = 12

# Number of iterations of MP algorithm to run for each frame
n_iter = 500

# Covariance of Gaussian filter (this needs to be determined manually for each dataset)
cov = np.diag([3.0, 3.0, 1.0])


def main():

    print('Launching processes to compute MP components...')
    mps = mp_parallel.compute_mps(
        reader_factory=io.VivekReaderFactory(in_fpath),
        t_start=t_start, 
        t_stop=t_stop, 
        cov=cov, 
        n_iter=n_iter, 
        n_procs=n_procs
    )

    print(f'Complete. Writing results to {out_fpath}...')
    with io.MPWriter(out_fpath) as writer:
        writer.write(mps, t_start, t_stop)

    print('Done.')
    
    
if __name__ == '__main__':
    main()