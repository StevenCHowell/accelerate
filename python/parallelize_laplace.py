
# coding: utf-8

# Code adapted from NVIDIA OpenACC example: laplace2d-kernels.c
#
# Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import time
import pickle

from math import fabs

from numba import jit, njit, guvectorize


# N values to iterate over
pow_max = 13
N = [2 ** val for val in range(2, pow_max)]


# NumPy Code
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


# Adding numba jit decorator
@jit(['float32[:,:], float32[:,:]',
      'float64[:,:], float64[:,:]'],
      nopython=True)
def update_a_jit(A, Anew):
    error = 0.0
    n, m = A.shape

    for j in range(1, n-1):
        for i in range(1, m-1):
            Anew[j, i] = 0.25 * (A[j, i+1] + A[j, i-1] + A[j-1, i] + A[j+1, i])
            error = max(error, abs(Anew[j, i] - A[j, i]))

    return Anew, error


# this did not seem to make much difference
@jit(['float32[:,:], float32[:,:]',
      'float64[:,:], float64[:,:]'],
      nopython=True)
def a_to_anew_jit(A, Anew):
    error = 0.0
    n, m = A.shape

    for j in range(1, n-1):
        for i in range(1, m-1):
            A[j, i] = Anew[j, i]

    return A


def jit_laplace(n=4096, m=4096, iteration_max = 1000, debug=False):

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
        Anew, error = update_a_jit(A, Anew)
        # A[:] = Anew[:]
        A = a_to_anew_jit(A, Anew)  # this was essentially the same as the line above

        if debug and not iteration % 100:
            print("{:5d}, {:0.6f}\n".format(iteration, error))

        iteration += 1

    toc = time.time() - tic

    if debug: print(" total: {} s\n".format(toc))

    return A




# Implementing `guvectorize`

# CPU Vectorization
@guvectorize(['float32[:,:], float32[:,:]',
              'float64[:,:], float64[:,:]'],
             '(n,m)->(n,m)', nopython=True,
             target='cpu')
def update_a_serial0(A, Anew):
    n, m = A.shape
    for j in range(1, n-1):
        for i in range(1, m-1):
            Anew[j, i] = 0.25 * (A[j, i+1] + A[j, i-1] + A[j-1, i] + A[j+1, i])


@guvectorize(['float32[:,:], float32[:,:], float32[:]',
              'float64[:,:], float64[:,:], float64[:]'],
             '(n,m)->(n,m),()', nopython=True, target='cpu')
def update_a_serial(A, Anew, error):
    n, m = A.shape
    for j in range(1, n-1):
        for i in range(1, m-1):
            Anew[j, i] = 0.25 * (A[j, i+1] + A[j, i-1] + A[j-1, i] + A[j+1, i])

            diff = abs(Anew[j, i] - A[j, i])
            if diff > error[0]:
                error[0] = diff


def cpu_laplace(n=4096, m=4096, iteration_max = 1000, debug=False):

    pi = np.pi
    tol = 1.0e-5
    error = np.ones(1)

    A = np.zeros((n, m))

    j_range = np.arange(n)
    y0 = np.sin(pi * j_range / (n - 1))
    A[:, 0]  = y0
    A[:, -1] = y0 * np.exp(-pi)

    if debug: print('Jacobi relaxation Calculation: {} x {} mesh\n'.format(n, m))

    tic = time.time()
    iteration = 0

    Anew = np.copy(A)
    while error[0] > tol and iteration < iteration_max:
        error[0] = 0.0
        Anew, error = update_a_serial(A, Anew, error)
        A[:] = Anew[:]

        if debug and not iteration % 100:
            print("{:5d}, {:0.6f}\n".format(iteration, error[0]))

        iteration += 1

    toc = time.time() - tic

    if debug: print(" total: {} s\n".format(toc))

    return A


# parallel CPU vectorization
@guvectorize(['float32[:,:], float32[:,:], float32[:]',
              'float64[:,:], float64[:,:], float64[:]'],
             '(n,m)->(n,m),()', nopython=True, target='parallel')
def update_a_parallel(A, Anew, error):
    n, m = A.shape
    for j in range(1, n-1):
        for i in range(1, m-1):
            Anew[j, i] = 0.25 * (A[j, i+1] + A[j, i-1] + A[j-1, i] + A[j+1, i])

            diff = abs(Anew[j, i] - A[j, i])
            if diff > error[0]:
                error[0] = diff


def par_laplace(n=4096, m=4096, iteration_max = 1000, debug=False):
    pi = np.pi
    tol = 1.0e-5
    error = np.ones(1)

    A = np.zeros((n, m))

    j_range = np.arange(n)
    y0 = np.sin(pi * j_range / (n - 1))
    A[:, 0]  = y0
    A[:, -1] = y0 * np.exp(-pi)

    if debug: print('Jacobi relaxation Calculation: {} x {} mesh\n'.format(n, m))

    tic = time.time()
    iteration = 0

    Anew = np.copy(A)
    while error[0] > tol and iteration < iteration_max:
        error[0] = 0.0
        Anew, error = update_a_parallel(A, Anew, error)
        A[:] = Anew[:]

        if debug and not iteration % 100:
            print("{:5d}, {:0.6f}\n".format(iteration, error[0]))

        iteration += 1

    toc = time.time() - tic

    if debug: print(" total: {} s\n".format(toc))

    return A


# ### CUDA vectorization
@guvectorize(['float32[:,:], float32[:,:]',
              'float64[:,:], float64[:,:]'],
             '(n,m)->(n,m)', nopython=True, target='cuda')
def update_a_cuda(A, Anew):
    n, m = A.shape
    for j in range(1, n-1):
        for i in range(1, m-1):
            Anew[j, i] = 0.25 * (A[j, i+1] + A[j, i-1] + A[j-1, i] + A[j+1, i])


@guvectorize(['float32[:,:], float32[:,:], float32[:]',
              'float64[:,:], float64[:,:], float64[:]'],
             '(n,m),(n,m)->()', nopython=True, target='cuda')
def update_error_cuda(A, Anew, error):
    n, m = A.shape
    for j in range(1, n-1):
        for i in range(1, m-1):
            diff = abs(Anew[j, i] - A[j, i])
            if diff > error[0]:
                error[0] = diff


def cuda_laplace(n=4096, m=4096, iteration_max = 1000, debug=False):

    dtype=np.float32

    pi = np.pi
    tol = 1.0e-5
    error = np.ones(1)

    A = np.zeros((n, m), dtype=dtype)

    j_range = np.arange(n)
    y0 = np.sin(pi * j_range / (n - 1))
    A[:, 0]  = y0
    A[:, -1] = y0 * np.exp(-pi)

    if debug: print('Jacobi relaxation Calculation: {} x {} mesh\n'.format(n, m))

    tic = time.time()
    iteration = 0

    Anew = np.copy(A)
    while error[0] > tol and iteration < iteration_max:
        error[0] = 0.0
        res = update_a_cuda(A)
        Anew[1:-1, 1:-1] = res[1:-1, 1:-1]
        error = update_error_cuda(A, Anew)
        A[:] = Anew[:]

        if debug and not iteration % 100:
            print("{:5d}, {:0.6f}\n".format(iteration, error[0]))

        iteration += 1

    toc = time.time() - tic

    if debug: print(" total: {} s\n".format(toc))

    return A



# Only vectorizing the A-matrix update on the GPU is much slower,
# likely because of data transfer after each iteration.
# The following code vectorizes the entire while loop on the GPU.
@guvectorize([
        'float32[:,:], float32[:,:]',
        'float64[:,:], float64[:,:]',
    ], '(n,m)->(n,m)', target='cuda', nopython=True)
def iterate_a(A, Anew):
    n, m = A.shape
    tol = 1.0e-5
    iteration_max = 1000
    error = 1.0
    iteration = 0

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


def cuda_laplace_v2(n=4096, m=4096, debug=False):

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


# compare serial vectorization of the entire while loop
@guvectorize([
        'float32[:,:], float32[:,:]',
        'float64[:,:], float64[:,:]',
    ], '(n,m)->(n,m)', target='cpu', nopython=True)
def cpu_iterate_a(A, Anew):
    n, m = A.shape
    tol = 1.0e-5
    iteration_max = 1000
    error = 1.0
    iteration = 0

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


def cpu_laplace_v2(n=4096, m=4096, debug=False):

    dtype=np.float32

    pi = np.pi

    A = np.zeros((n, m), dtype=dtype)

    j_range = np.arange(n)
    y0 = np.sin(pi * j_range / (n - 1))
    A[:, 0]  = y0
    A[:, -1] = y0 * np.exp(-pi)

    if debug: print('Jacobi relaxation Calculation: {} x {} mesh\n'.format(n, m))

    tic = time.time()
    cpu_iterate_a(A)
    toc = time.time() - tic

    if debug: print(" total: {} s\n".format(toc))

    return A


# compare parallel vectorization of the entire while loop
@guvectorize([
        'float32[:,:], float32[:,:]',
        'float64[:,:], float64[:,:]',
    ], '(n,m)->(n,m)', target='parallel', nopython=True)
def par_iterate_a(A, Anew):
    n, m = A.shape
    tol = 1.0e-5
    iteration_max = 1000
    error = 1.0
    iteration = 0

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


def par_laplace_v2(n=4096, m=4096, debug=False):

    dtype=np.float32

    pi = np.pi

    A = np.zeros((n, m), dtype=dtype)

    j_range = np.arange(n)
    y0 = np.sin(pi * j_range / (n - 1))
    A[:, 0]  = y0
    A[:, -1] = y0 * np.exp(-pi)

    if debug: print('Jacobi relaxation Calculation: {} x {} mesh\n'.format(n, m))

    tic = time.time()
    par_iterate_a(A)
    toc = time.time() - tic

    if debug: print(" total: {} s\n".format(toc))

    return A


n = 4
res = np_laplace(n, n, debug=True)
res = jit_laplace(n, n, debug=True)
res = cpu_laplace(n, n, debug=True)
res = par_laplace(n, n, debug=True)
res = cuda_laplace(n, n, debug=True)
res = cuda_laplace_v2(n, n, debug=True)
res = cpu_laplace_v2(n, n, debug=True)
res = par_laplace_v2(n, n, debug=True)

np_res = {}
jit_res = {}
cpu_res = {}
par_res = {}
cuda_res = {}
cuda_res_v2 = {}
cpu_res_v2 = {}
par_res_v2 = {}


'''
# profile np_laplace
for n in N:
    res = %timeit -o -n 3 np_laplace(n, n)
    np_res[n] = res
    pickle.dump(np_res, open('np_times.p', 'wb'))


# profile jit version of laplace
for n in N:
    res = %timeit -o -n 3 jit_laplace(n, n)
    jit_res[n] = res
    pickle.dump(jit_res, open('jit_times.p', 'wb'))


# profile serial vectorization of laplace update
for n in N:
    res = %timeit -o -n 3 cpu_laplace(n, n)
    cpu_res[n] = res
    pickle.dump(cpu_res, open('cpu_times.p', 'wb'))


# profile parallel vectorization of laplace update
for n in N:
    res = %timeit -o -n 3 par_laplace(n, n)
    par_res[n] = res
    pickle.dump(par_res, open('par_times.p', 'wb'))


# profile CUDA vectorization of laplace update
for n in N:
    res = %timeit -o -n 3 cuda_laplace(n, n)
    cuda_res[n] = res
    pickle.dump(cuda_res, open('onsager_compute-0-0/cuda_times.p', 'wb'))


# profile CUDA vectorization of entire laplace
for n in N:
    res = %timeit -o -n 3 cuda_laplace_v2(n, n)
    cuda_res_v2[n] = res
    pickle.dump(cuda_res_v2, open('onsager_compute-0-0/cuda_times_v2.p', 'wb'))


# profile serial vectorization of entire laplace
for n in N:
    res = %timeit -o -n 3 cpu_laplace_v2(n, n)
    cpu_res_v2[n] = res
    pickle.dump(cpu_res_v2, open('cpu_times_v2.p', 'wb'))

# profile parallel vectorization of entire laplace
for n in N:
    res = %timeit -o -n 3 par_laplace_v2(n, n)
    par_res_v2[n] = res
    pickle.dump(par_res_v2, open('par_times_v2.p', 'wb'))


'''
