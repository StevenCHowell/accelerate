import os
import time

import numpy as np
from scipy.spatial.distance import pdist

import numba
from numba import jit, njit, vectorize, guvectorize

'''
import bokeh.plotting
import bokeh.io
bokeh.io.output_file('pr_script.html')


from bokeh.palettes import Dark2_7 as palette
'''

def calc_pr_hist(coor):
    # calculate the euclidean distances
    dist = pdist(coor)

    # bin the distances into P(r)
    r_max = dist.max()

    n_bins = np.ceil(r_max).astype(int) + 1
    bin_edges = np.arange(n_bins + 1) - 0.5
    pr = np.empty([n_bins, 2])
    pr[:, 0] = np.arange(n_bins) + 1
    pr[:, 1], _ = np.histogram(dist, bin_edges)
    return pr


def calc_pr_bin(coor):
    # calculate the euclidean distances
    dist = pdist(coor)

    # bin the distances into P(r)
    r_max = dist.max()

    n_bins = np.ceil(r_max).astype(int) + 1
    pr = np.empty([n_bins, 2], dtype=np.int)
    pr[:, 0] = np.arange(n_bins) + 1
    int_dist = np.round(dist).astype(np.int)
    pr[:, 1] = np.bincount(int_dist)
    return pr


@jit(['int64[:], int64[:]',
      'float32[:], int64[:]',
      'float64[:], int64[:]'],
     nopython=True)
def jit_hist(distances, pr):
    for dist in distances:
        pr[int(round(dist))] += 1


def jit_calc_pr(coor):
    # calculate the euclidean distances
    dist = pdist(coor)

    # bin the distances into P(r)
    r_max = dist.max()
    n_bins = np.round(r_max).astype(int) + 1
    pr = np.zeros([n_bins, 2], dtype=np.int)
    pr[:, 0] = np.arange(n_bins) + 1
    jit_hist(dist, pr[:, 1])
    return pr


coor_fname = 'nl1_nrx1b.coor'
coor = np.loadtxt(coor_fname, delimiter=',')
out_fname = '{}.pr'.format(os.path.splitext(coor_fname))

pr2 = jit_calc_pr(coor)
pr1 = calc_pr_bin(coor)
pr0 = calc_pr_hist(coor)

'''
p = bokeh.plotting.figure()
p.square(pr0[:, 0], pr0[:, 1], color=palette[0], legend='np.histogram')
p.circle(pr1[:, 0], pr1[:, 1] * 2, color=palette[1], legend='np.bincount')
p.circle(pr2[:, 0], pr2[:, 1] * 3, color=palette[2], legend='numba.jit')

bokeh.io.show(p)
'''

# In[ ]:

"""
@numba.jit(['int64[:], int64[:]',
      'float32[:], int64[:]',
      'float64[:], int64[:]'],
     nopython=True)
def jit_hist(distances, pr):
    for dist in distances:
        pr[round(dist)] += 1


def calc_pr_numba(coor):
    '''
    calculate P(r) from an array of coordinates
    when written, this was twice as fast as python method
    '''
    # calculate the euclidean distances
    dist = pdist(coor)

    # bin the distances into P(r)
    r_max = dist.max()
    n_bins = np.round(r_max).astype(int) + 1
    pr = np.zeros([n_bins, 2], dtype=np.int)
    pr[:, 0] = np.arange(n_bins) + 1
    jit_histogram(dist, pr[:, 1])

    return pr
"""
