"""Script for running matching pursuit (MP) on video, using parallelization"""

import argparse
import functools
from multiprocessing import Pool

import numpy as np
from scipy.io import savemat

from otimage import io, imagerep


# Default number of processes used
DEFAULT_NUM_PROCS = 12

# Default number of iterations
DEFAULT_NUM_ITER = 500

# Covariance values for each dimension

# Zimmer
COV_DIAG = [5.0, 5.0, 5.0]

# Vivek
#COV_DIAG = [3.0, 3.0, 1.0]


def positive_int(val_str):
    """Parse positive integer from string."""

    val_int = int(val_str)
    if val_int > 0:
        return val_int
    else:
        raise argparse.ArgumentTypeError(f'Invalid value: "{val_str}"')


def non_negative_int(val_str):
    """Parse non-negative integer from string."""

    val_int = int(val_str)
    if val_int >= 0:
        return val_int
    else:
        raise argparse.ArgumentTypeError(f'Invalid value: "{val_str}"')


def parse_args():
    """Parse arguments from command-line."""

    parser = argparse.ArgumentParser(description='Compute MPs for dataset')

    parser.add_argument('--input', '-i', help='path to input file')
    parser.add_argument('--output', '-o', help='path to output file')
    parser.add_argument(
        '--dtype', '-d', 
        choices=['synthetic', 'zimmer', 'vivek'],
        help='dataset type'
    )
    parser.add_argument(
        '--start', '-s', 
        default=None,
        type=non_negative_int, 
        help='frame to start at'
    )
    parser.add_argument(
        '--end', '-e', 
        default=None,
        type=non_negative_int, 
        help='frame to end at'
    )
    parser.add_argument(
        '--procs', '-p', 
        default=DEFAULT_NUM_PROCS,
        type=positive_int, 
        help='number of processes'
    )
    parser.add_argument(
        '--niter', '-n', 
        default=DEFAULT_NUM_ITER,
        type=positive_int, 
        help='number of iterations'
    )

    return parser.parse_args()


def get_reader(fpath, dtype):
    """Get io.WormDataReader object of specific datatype for given path.

    Args:
        fpath (str): Path to file containing data
        dtype (str): Type of data in file (either 'synthetic', 'zimmer',
            or 'vivek')

    Returns:
        io.WormDataReader object for dataset

    """

    if dtype == 'synthetic':
        return io.SyntheticReader(fpath)
    elif dtype == 'zimmer':
        return io.ZimmerReader(fpath)
    elif dtype == 'vivek':
        return io.VivekReader(fpath)
    else:
        raise ValueError(f'Not valid dataset type: "dtype"')


def get_limits(args):
    """Determine start and stop frames from args.

    Args:
        args (argparse.Namespace): Argument namespace containing the following
            fields:
                input (str): Path to input file
                dtype (str): Data type ('synthetic', 'zimmer', or 'vivek')
                start (int or None): Start frame
                end (int or None): Stop frame

    Returns:
        (t_start, t_stop): Start and stop frames (both ints)
    
    """

    with get_reader(args.input, args.dtype) as reader:
        num_frames = reader.num_frames

    t_start = args.start if args.start is not None else 0
    t_stop = args.end if args.end is not None else num_frames

    return (t_start, t_stop)


def get_chunks(t_start, t_stop, n_chunks):
    """Group frame indices into given number of 'chunks'.

    Args:
        t_start (int): Frame index to start at (inclusive)
        t_stop (int): Frame index to stop at (exclusive)
        n_chunks (int): Number of chunks

    Returns:
        List of 2-tuples containing (start, stop) for each chunk.

    """

    # Validate input
    if t_stop <= t_start:
        raise ValueError('Start frame not before stop frame')
    if n_chunks <= 0:
        raise ValueError('Number of chunks not positive int')
    if n_chunks > (t_stop - t_start):
        raise ValueError('More chunks than frames')

    # Determine size of chunks
    sz = (t_stop - t_start) // n_chunks

    # First n-1 chunks
    chunks = []
    for k in range(n_chunks - 1):
        chunks.append((t_start + k * sz, t_start + (k + 1) * sz))

    # Final chunk
    chunks.append((t_start + (n_chunks - 1) * sz, t_stop))

    return chunks


def get_mps(rng, fpath, dtype, cov, n_iter):
    """Run Gaussian MP algorithm on set of frames from data file.

    This function is meant to be executed by a single worker process. It reads
    its set of frames from the data file, runs the Greedy Matching Pursuit
    algorithm (Elad, 2014) on each frame with the specified parameters, and
    returns the results for each frame in a list.

    Args:
        rng (2-tuple): Range of frames to run MP on. The algorithm will be run
            on all frames from rng[0] (inclusive) to rng[1] (exclusive).
        fpath (str): Path to file containing dataset.
        dtype (str): String indicating type of dataset. Either 'synthetic',
            'zimmer', or 'vivek'.
        cov (np.ndarray): Covariance matrix of Gaussian filter used in
            algorithm.
        n_iter (int): Number of iterations to run algorithm for.

    Returns:
        list of imagerep.ImageMP objects: MP representations for images

    """

    t_start, t_stop = rng

    with get_reader(fpath, dtype) as reader:

        mps = []
        
        for t in range(t_start, t_stop):
            
            img = reader.get_frame(t)
            mp, _ = imagerep.mp_gaussian(img, cov, n_iter)
            mps.append(mp)
            
    return mps 


def main():

    args = parse_args()
    t_start, t_stop = get_limits(args)

    # Covariance matrix for MP components
    cov = np.diag(COV_DIAG)

    # Split frames into chunks for each process
    chunks = get_chunks(t_start, t_stop, args.procs)

    # Run MP algorithm on frames across chunks
    print('Launching processes to compute MP components...')
    with Pool(processes=args.procs) as p:
        _get_mps = functools.partial(
            get_mps,
            fpath=args.input,
            dtype=args.dtype,
            cov=cov, 
            n_iter=args.niter
        )
        results = p.map(_get_mps, chunks)
    
    # Compile chunk results into list of ImageMP objects
    mps = [x for r in results for x in r]
        
    # Write MPs to MAT file
    print(f'Complete. Writing results to {args.output}...')
    with io.MPWriter(args.output) as writer:
        writer.write(mps, t_start, t_stop)

if __name__ == '__main__':
    main()
