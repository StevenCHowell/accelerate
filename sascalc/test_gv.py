import time
import numpy as np


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Set the golden vectors
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_gv(n_gv):

    # setup GV
    if not isinstance(n_gv, (int, np.integer)):
        print('The number of golden vectors should be an integer. ',
              'It will be changed from {} to {}.'.format(n_gv, int(n_gv)))
        n_gv = int(n_gv)
    if not n_gv % 2:
        print('The number of golden vectors should be odd. ',
              'It will be changed from {} to {}.'.format(n_gv, n_gv + 1))
        n_gv += 1

    gv = np.empty((int(n_gv), 3))
    n_gv = float(n_gv)  # to prevent division problems in python 2

    phi_inv = 2.0 / (1 + np.sqrt(5.0))  # golden ratio

    rank = n_gv / 2.0

    i = np.arange(-rank, rank)
    sin_theta = np.cos(np.arcsin(2.0 * i / n_gv))
    cos_theta = 2.0 * i / n_gv
    phi = 2 * np.pi * i * phi_inv

    gv[:, 0] = sin_theta * np.cos(phi)
    gv[:, 1] = sin_theta * np.sin(phi)
    gv[:, 2] = cos_theta

    # loop method
    '''
    for i in np.arange(-rank, rank):
        sin_theta = np.cos(np.arcsin(2.0 * i / n_gv))
        cos_theta = 2.0 * i / n_gv
        phi = 2 * np.pi * i * phi_inv

        i_gv = int(i + rank)
        gv[i_gv, 0] = sin_theta * np.cos(phi)
        gv[i_gv, 1] = sin_theta * np.sin(phi)
        gv[i_gv, 2] = cos_theta
    '''

    return gv


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Golden vector internal calculator
#   for single item and single frame using fixed method
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calc_iq(coor, b, n_atoms, n_q, dq, n_gv):
    # locals
    gv = get_gv(n_gv)
    iq = np.empty(n_q)

    tic = time.time()

    # summation
    for i in range(n_q):
        q_mag = i * dq
        for j in range(n_gv):
            qx = q_mag * gv[j, 0]
            qy = q_mag * gv[j, 1]
            qz = q_mag * gv[j, 2]
            int_real = 0.0
            int_mag = 0.0
            for k in range(n_atoms):
                int_mag += b[k] * np.sin(qx * coor[k, 0] +
                                         qy * coor[k, 1] +
                                         qz * coor[k, 2])
                int_real += b[k] * np.cos(qx * coor[k, 0] +
                                          qy * coor[k, 1] +
                                          qz * coor[k, 2])
            iq[i] += int_real * int_real + int_mag * int_mag
        iq[i] /= n_gv


    # clean up
    toc = time.time() - tic
    print("TIME = {}".format(toc))

    return iq


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   main
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    # create dummy coordinates and Bs
    # which is a linear molecule along X-axis with 10 atoms

    n_atoms = 1000000
    n_atoms = 1000
    coor = np.zeros((n_atoms, 3))
    coor[:, 0] = np.arange(n_atoms)

    b = np.ones(n_atoms)

    # loop version
    '''
    for i in range(n_atoms):
        coor[i, 0] = i_atom
        coor[i, 1] = 0.0
        coor[i, 2] = 0.0

        b[i_atom] = 1.0
    '''

    # Iq stuff
    n_q = 10
    dq = 0.1
    n_gv = 31

    # calculate Iq
    iq = calc_iq(coor, b, n_atoms, n_q, dq, n_gv)

    # print Iq
    print("q \t\t I(q) \t\t I(q)/I(0) \n")
    for i in range(n_q):
        print("{} \t {}       \t {}\n".format(i * dq, iq[i], iq[i]/iq[0]))
        # print("{:8.3f} {:8.3f} {:8.3f}\n".format(i * dq, iq[i]/iq[0], iq[i]))



