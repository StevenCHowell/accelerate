import numpy as np
import time
import pickle
from numba import guvectorize, jit, njit
from math import fabs
pow_max = 13
N = [2 ** val for val in range(2, pow_max)]


def np_laplace(n=4096, m=4096, iteration_max = 1000, debug=False):

    pi = np.pi
    tol = 1.0e-5
    error = 1.0

    A = np.zeros((n, m))

    j_range = np.arange(n)
    y0 = np.sin(pi * j_range / (n - 1))
    A[:, 0]  = y0
    A[:, -1] = y0 * np.exp(-pi)

    if debug: print('Jacobi relaxation Calculation: {} x {} mesh\n'.format(n, m))

    tic = time.time()
    iteration = 0

    Anew = np.copy(A)

    while error > tol and iteration < iteration_max:
        error = 0.0
        Anew[1:-1, 1:-1] = 0.25 * (A[1:-1, 2:] + A[1:-1, :-2] + A[2:, 1:-1] + A[:-2, 1:-1])
        diff = np.abs(Anew[1:-1, 1:-1] - A[1:-1, 1:-1])
        error_new = np.max(diff)
        if error_new > error:
            error = error_new
        A[:] = Anew[:]

        if debug and not iteration % 100:
            print("{:5d}, {:0.6f}\n".format(iteration, error))

        iteration += 1

    toc = time.time() - tic

    if debug: print(" total: {} s\n".format(toc))

    return A


@guvectorize([
        'float32[:,:], float32[:,:]',
        'float64[:,:], float64[:,:]',
    ], '(n,m)->(n,m)', target='cpu', nopython=True)
def iterate_a(A, Anew):
    n, m = A.shape
    tol = 1.0e-5
    iteration_max = 1000
    error = 1.0
    iteration = 0

    for j in range(n):
        for i in range(m):
            Anew[j, i] = 0.0

    # main loop
    while error > tol and iteration < iteration_max:
        error = 0.0
        for j in range(1, n-1):
            for i in range(1, m-1):
                Anew[j, i] = 0.25 * (A[j, i+1] + A[j, i-1] + A[j-1, i] + A[j+1, i])
                diff = abs(Anew[j, i] - A[j, i])
                if diff > error:
                    error = diff

        for j in range(1, n-1):
            for i in range(1, m-1):
                A[j, i] = Anew[j, i]

        iteration += 1


def cuda_laplace(n=4096, m=4096, debug=False):

    dtype=np.float32

    pi = np.pi

    A = np.zeros((n, m), dtype=dtype)

    j_range = np.arange(n)
    y0 = np.sin(pi * j_range / (n - 1))
    A[:, 0]  = y0
    A[:, -1] = y0 * np.exp(-pi)

    if debug: print('Jacobi relaxation Calculation: {} x {} mesh\n'.format(n, m))

    tic = time.time()
    iterate_a(A)
    toc = time.time() - tic

    if debug: print(" total: {} s\n".format(toc))

    return A

n_res = np_laplace(4, 4, debug=True)
c_res = cuda_laplace(4, 4, debug=True)
print(n_res)
print(c_res)
print(np.allclose(n_res, c_res))
