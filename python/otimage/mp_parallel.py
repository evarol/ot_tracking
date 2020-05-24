"""Functions for running matching pursuit (MP) on video using parallelization"""

import functools
from multiprocessing import Pool

from otimage import io, imagerep


def _get_reader(fpath, dtype):
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


def _get_chunks(t_start, t_stop, n_chunks):
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


def _get_mps(rng, fpath, dtype, cov, n_iter):
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

    with _get_reader(fpath, dtype) as reader:

        mps = []
        
        for t in range(t_start, t_stop):
            
            img = reader.get_frame(t)
            mp, _ = imagerep.mp_gaussian(img, cov, n_iter)
            mps.append(mp)
            
    return mps 


def compute_mps(in_fpath, dtype, t_start, t_stop, cov, n_iter, n_procs):
    
    # Split frames into chunks for each process
    chunks = _get_chunks(t_start, t_stop, n_procs)

    # Run MP algorithm on frames across chunks
    with Pool(processes=n_procs) as p:
        get_mps_chunk = functools.partial(
            _get_mps,
            fpath=in_fpath,
            dtype=dtype,
            cov=cov, 
            n_iter=n_iter
        )
        results = p.map(get_mps_chunk, chunks)
    
    # Compile chunk results into list of ImageMP objects
    return [x for r in results for x in r]