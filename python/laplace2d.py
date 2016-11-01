#!/usr/bin/env python
#
# Author:  Steven C. Howell
# Purpose: calculating the Laplace equation using Python with CUDA
# Created: 1 November 2016
#
#0000000011111111112222222222333333333344444444445555555555666666666677777777778
#2345678901234567890123456789012345678901234567890123456789012345678901234567890

'''
Code adapted from NVIDIA OpenACC example: laplace2d-kernels.c

Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import time
import numpy as np

# add check for gtx vs tesla/quadro card then set float single/double precision


def main():

    n = 32
    m = 32
    iteration_max = 1000

    pi = np.pi
    tol = 1.0e-5
    error = 1.0

    A = np.zeros((n, m))
    '''
    Anew = np.zeros((n, m))
    y0 = np.zeros(n)

    # set boundary conditions
    A[0, :]  = 0.0
    A[-1, :] = 0.0
    '''

    '''
    for j in range(n):
        y0[j] = np.sin(pi * j / (n - 1))
        A[j, 0] = y0[j]
        A[j, -1] = y0[j] * np.exp(-pi)
    '''
    j_range = np.arange(n)
    y0 = np.sin(pi * j_range / (n - 1))
    A[:, 0]  = y0
    A[:, -1] = y0 * np.exp(-pi)

    '''
    #if _OPENACC
        acc_init(acc_device_nvidia)
    #endif
    '''

    print('Jacobi relaxation Calculation: {} x {} mesh\n'.format(n, m))

    tic = time.time()
    iteration = 0

    #pragma omp parallel for shared(Anew)
    '''
    for i in range(m):
        # enforce the boundary conditions
        Anew[0, i]  = 0.0
        Anew[-1, i] = 0.0
    '''

    #pragma omp parallel for shared(Anew)
    '''
    for j in range(n):
        Anew[j, 0]  = y0[j]
        Anew[j, -1] = y0[j] * np.exp(-pi)
    '''
    Anew = np.copy(A)


    #pragma acc data copy(A), create(Anew)
    while error > tol and iteration < iteration_max:

        error = 0.0

        #pragma omp parallel for shared(m, n, Anew, A)
        #pragma acc kernels loop gang(32), vector(16)
        for j in range(1, n-1):

            #pragma acc loop gang(16), vector(32)
            for i in range(1, m-1):

                Anew[j, i] = 0.25 * ( A[j, i+1] + A[j, i-1] +
                                      A[j-1, i] + A[j+1, i])
                error = np.max([error, np.abs(Anew[j, i] - A[j, i])])



        #pragma omp parallel for shared(m, n, Anew, A)
        #pragma acc kernels loop
        for j in range(1, n-1):

            #pragma acc loop gang(16), vector(32)
            for i in range(1, m-1):

                A[j, i] = Anew[j, i]



        if not iteration % 100:
            print("{:5d}, {:0.6f}\n".format(iteration, error))

        iteration += 1


    toc = time.time() - tic

    print(" total: {} s\n".format(toc))


if __name__ == "__main__":
    main()