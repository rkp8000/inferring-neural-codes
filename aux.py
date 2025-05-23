# miscellaneous useful functions and classes
import h5py
import numpy as np
import os
from scipy.linalg import hankel
import string

cc = np.concatenate

Dataset = h5py._hl.dataset.Dataset
Group = h5py._hl.group.Group
Reference = h5py.h5r.Reference


class Generic(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v

def rand_string(n):
    return ''.join(np.random.choice(list(string.ascii_letters + string.digits), n))


def c_tile(x, n):
    """Create tiled matrix where each of n cols is x."""
    return np.tile(x.flatten()[:, None], (1, n))


def r_tile(x, n):
    """Create tiled matrix where each of n rows is x."""
    return np.tile(x.flatten()[None, :], (n, 1))


def get_idx(z, z_0, dz, l):
    """
    Return closest valid integer index of continuous value.
    
    :param z: continuous value
    :param z_0: min continuous value (counter start point)
    :param dz: 
    """
    try:
        z[0]
    except:
        z = np.array([z]).astype(float)
        
    int_repr = np.round((z-z_0)/dz).astype(int)
    return np.clip(int_repr, 0, l-1)


def get_seg(x, min_gap):
    # get segments, and start/stop bounds
    included = (x).astype(int)
    change = np.diff(cc([[0], included, [0]]))
    start = (change == 1).nonzero()[0]
    stop = (change == -1).nonzero()[0]
    
    mask = np.ones((len(start), 2), dtype=bool)
    
    for cseg in range(len(stop)-1):
        if (start[cseg+1] - stop[cseg]) < min_gap:
            mask[cseg+1, 0] = False
            mask[cseg, 1] = False
            
    start = start[mask[:, 0]]
    stop = stop[mask[:, 1]]
    
    seg = [x[start_:stop_] for start_, stop_ in zip(start, stop)]
    bds = np.array(list(zip(start, stop)))  # M x 2 array of start, stop idxs
    
    return seg, bds


def split(x, istarts, iends):
    return [x[istart:iend] for istart, iend in zip(istarts, iends)]


def get_outliers(x, ndev=3):
    med = np.median(x)
    med_abs_dev = np.median(np.abs(x - med))
    
    return np.abs(x - med) > (ndev*med_abs_dev)


def mv_avg(t, x, wdw):
    # return symmetric moving average of x with wdw s
    x_avg = np.nan * np.zeros(x.shape)
    for it, t_ in enumerate(t):
        mt = ((t_- wdw/2) <= t) & (t < (t_ + wdw/2))
        x_avg[it] = np.nanmean(x[mt])
    return x_avg


def zscore(x):
    return (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)


def load_npy(file_name):
    return np.load(file_name, allow_pickle=True)[0]


def align_t(ts, xs, dt, t_min=None, t_max=None, tbase=None):
    """
    Align variable length/frame-rate timestamps and corresponding signal to common base time vector.
    (Slow implementation but works.)
    """
    
    if t_min is None:
        t_min = np.min(cc(ts))
    
    if t_max is None:
        t_max = np.max(cc(ts))
    
    # create base time vector
    if tbase is None:
        tbase = np.arange(t_min, t_max+dt, dt)
    
    # resample xs to align with base time vector
    xs_aligned = []
    
    for t, x in zip(ts, xs):
        
        x_aligned = np.nan * tbase
        for ct, t_0 in enumerate(tbase):
            t_1 = t_0 + dt
            if t_0 < t[-1]:
                x_aligned[ct] = np.nanmean(x[(t_0 <= t) & (t < t_1)])
        
        xs_aligned.append(x_aligned)
    
    return tbase, xs_aligned


def loadmat_h5(file_name):
    '''Loadmat equivalent for -v7.3 or greater .mat files, which break scipy.io.loadmat'''
    
    def deref_s(s, f, verbose=False):  # dereference struct
        keys = [k for k in s.keys() if k != '#refs#']
        
        if verbose:
            print(f'\nStruct, keys = {keys}')

        d = {}

        for k in keys:
            v = s[k]

            if isinstance(v, Group):  # struct
                d[k] = deref_s(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and v.size == 0:  # empty dataset
                d[k] = np.zeros(v.shape).T
            elif isinstance(v, Dataset) and isinstance(np.array(v).flat[0], Reference):  # cell
                d[k] = deref_c(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and v.dtype == 'uint16':
                d[k] = ''.join(np.array(v).view('S2').flatten().astype(str))
                if verbose:
                    print(f'String, chars = {d[k]}')
            elif isinstance(v, Dataset):  # numerical array
                d[k] = np.array(v).T
                if verbose:
                    print(f'Numerical array, shape = {d[k].shape}')

        return d

    def deref_c(c, f, verbose=False):  # dereference cell
        n_v = c.size
        shape = c.shape

        if verbose:
            print(f'\nCell, shape = {shape}')

        a = np.zeros(n_v, dtype='O')

        for i in range(n_v):
            v = f['#refs#'][np.array(c).flat[i]]

            if isinstance(v, Group):  # struct
                a[i] = deref_s(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and v.size == 0:  # empty dataset
                d[k] = np.zeros(v.shape).T
            elif isinstance(v, Dataset) and isinstance(np.array(v).flat[0], Reference):  # cell
                a[i] = deref_c(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and v.dtype == 'uint16':
                a[i] = ''.join(np.array(v).view('S2').flatten().astype(str))
                if verbose:
                    print(f'String, chars = {a[i]}')
            elif isinstance(v, Dataset):  # numerical array
                a[i] = np.array(v).T
                if verbose:
                    print(f'Numerical array, shape = {a[i].shape}')

        return a.reshape(shape).T
    
    with h5py.File(file_name, 'r+') as f:
        d = deref_s(f, f)
        
    return d


def make_extended_predictor_matrix(vs, windows, order):
    """
    Make a predictor matrix that includes offsets at multiple time points.
    For example, if vs has 2 keys 'a' and 'b', windows is {'a': (-1, 1),
    'b': (-1, 2)}, and order = ['a', 'b'], then result rows will look like:
    
        [v['a'][t-1], v['a'][t], v['b'][t-1], v['b'][t], v['b'][t+1]]
        
    :param vs: dict of 1-D array of predictors
    :param windows: dict of (start, end) time point tuples, rel. to time point of 
        prediction, e.g., negative for time points before time point of
        prediction
    :param order: order to add predictors to final matrix in
    :return: extended predictor matrix, which has shape
        (n, (windows[0][1]-windows[0][0]) + (windows[1][1]-windows[1][0]) + ...)
    """
    if not np.all([w[1] - w[0] >= 0 for w in windows.values()]):
        raise ValueError('Windows must all be non-negative.')
        
    n = len(list(vs.values())[0])
    if not np.all([v.ndim == 1 and len(v) == n for v in vs.values()]):
        raise ValueError(
            'All values in "vs" must be 1-D arrays of the same length.')
        
    # make extended predictor array
    vs_extd = []
    
    # loop over predictor variables
    for key in order:
        
        start, end = windows[key]
        
        # make empty predictor matrix
        v_ = np.nan * np.zeros((n, end-start))

        # loop over offsets
        for col_ctr, offset in enumerate(range(start, end)):

            # fill in predictors offset by specified amount
            if offset < 0:
                v_[-offset:, col_ctr] = vs[key][:offset]
            elif offset == 0:
                v_[:, col_ctr] = vs[key][:]
            elif offset > 0:
                v_[:-offset, col_ctr] = vs[key][offset:]

        # add offset predictors to list
        vs_extd.append(v_)

    # return full predictor matrix
    return np.concatenate(vs_extd, axis=1)

def make_dsn_mat(x, nbak):
    return hankel(cc([np.zeros(nbak-1), x]))[:len(x), :nbak]
