import numba
import numpy as np
from scipy.spatial.distance import pdist

import platform
print('python: {}'.format(platform.python_version()))
print('numba: {}'.format(numba.__version__))
print('numpy: {}'.format(np.__version__))



@numba.jit(['int64[:], int64[:]',
            'float32[:], int64[:]',
            'float64[:], int64[:]'],
            nopython=True)
def jit_hist(distances, pr):
    for dist in distances:
        pr[round(dist)] += 1


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


n = 20000
coor = np.random.random(n*3).reshape((n, 3)) * 100

pr = jit_calc_pr(coor)

np.savetxt('test.pr', pr)
