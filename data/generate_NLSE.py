#!/usr/bin/env python

"""
generate_NLSE.py

Script for generating the 1D nonlinear Schrödinger dataset.
"""

import os

import numpy as np
from scipy.fftpack import fft, ifft
from scipy.stats import truncnorm
from scipy.signal import resample

from joblib import Parallel, delayed
import multiprocessing

import warnings
import argparse


def generateData(i, data_size, batch_size, l, out_mesh, tmax, out_tmesh, shift, kappa_mean, kappa_std):
    ### Generate data

    ### Uses ETDRK4 method for integrating stiff PDEs
    ### https://epubs.siam.org/doi/pdf/10.1137/S1064827502410633

    # np.random.seed()
    # print(np.random.rand())

    print('Batch ' + str(i+1) + ' of ' + str(int(data_size/batch_size)))

    kappa = truncnorm.rvs((-2 - kappa_mean) / kappa_std, (0 - kappa_mean) / kappa_std, 
                            loc=kappa_mean, scale=kappa_std, size=(batch_size, 1), 
                            random_state=np.random.RandomState())

    pool = 10
    print('Pooling: ' + str(pool))
    tpool = 20
    mesh = out_mesh * pool
    tmesh = out_tmesh * tpool
    dt = tmax/tmesh # time step

    k = 2*np.pi * np.fft.fftfreq(mesh, d=l/mesh)

    ## initial condition
    krange = ((out_mesh/8)*np.pi/l - 4 * np.pi/l) * np.random.rand(batch_size, 1) + 4 * np.pi/l
    envelope = np.exp(-1/(2*krange**2) * k**2)
    v0 = envelope * (np.random.normal(loc=0, scale=1.0, size=(batch_size, mesh)) 
                        + 1j*np.random.normal(loc=0, scale=1.0, size=(batch_size, mesh)))
    u0 = ifft(v0)
    u0 = np.sqrt(2 * mesh) * u0/np.expand_dims(np.linalg.norm(u0, axis=-1), axis=-1) # normalize
    v0 = fft(u0)

    ## differential equation
    L = -0.5j * k**2
    def N(v):
        u = ifft(v)
        return -1j * kappa * fft(np.abs(u)**2 * u)

    ## ETDRK4 method
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    contour_radius = 1
    M = 32
    r = contour_radius*np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    r = r.reshape(1, -1)
    r_contour = np.repeat(r, mesh, axis=0)

    LR = dt * L
    LR = np.expand_dims(LR, axis=-1)
    LR = np.repeat(LR, M, axis=-1)
    LR = LR + r_contour

    Q = dt*np.mean( (np.exp(LR/2.0)-1)/LR, axis=-1 )
    f1 = dt*np.mean( (-4.0-LR + np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3, axis=-1 )
    f2 = dt*np.mean( (2.0+LR + np.exp(LR)*(-2.0 + LR))/LR**3, axis=-1 )
    f3 = dt*np.mean( (-4.0-3.0*LR - LR**2 + np.exp(LR)*(4.0 - LR))/LR**3, axis=-1 )

    t = 0.0
    u = []
    v = v0
    tpool_num = 0
    
    # catch overflow warnings and rerun the data generation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter(action='ignore', category=FutureWarning)

        while t < tmax + shift - 1e-8:
            if t >= shift and tpool_num % tpool == 0: # exclude first 'shift' time
                up = resample(ifft(v), out_mesh, axis=-1)
                u.append(np.stack([np.real(up), np.imag(up)], axis=-2))

            Nv = N(v)
            a = E2 * v + Q * Nv
            Na = N(a)
            b = E2 * v + Q * Na
            Nb = N(b) 
            c = E2 * a + Q * (2.0*Nb - Nv)
            Nc = N(c)   
            v = E*v + Nv*f1 + 2.0*(Na + Nb)*f2 + Nc*f3

            t = t + dt
            tpool_num += 1

        if w:
            print('Rerunning...')
            print(w[-1])
            return generateData(i, data_size, batch_size, l, out_mesh, tmax, out_tmesh, shift, kappa_mean, kappa_std)

    return np.stack(u, axis=-2).astype(np.float32), kappa.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate nonlinear Schrödinger dataset.")
    parser.add_argument('-d', '--datasize', type=int, default=5000, 
        help="Size of generated dataset--should be a multiple of batch_size. Default: 5000")
    parser.add_argument('-b', '--batchsize', type=int, default=50, 
        help="Batch size for generating dataset in parallel--should divide data_size. Default: 50")
    parser.add_argument('-f', '--filename', type=str, default='NLSE_dataset.npz', 
        help="Path with file name ending in .npz where dataset is saved. Default: NLSE_dataset.npz")

    args = parser.parse_args()
    data_size = args.datasize
    batch_size = args.batchsize
    FILENAME = args.filename

    l = 8*np.pi # system size
    out_mesh = 256 # mesh
    tmax = 1*np.pi # max time
    out_tmesh = 256 # time mesh
    shift = 0 # shift time to exclude initial conditions, set to 0 to keep t = 0

    kappa_mean = -1
    kappa_std = 0.25

    num_cores = multiprocessing.cpu_count()
    print('Using ' + str(num_cores) + ' cores...')
    out_list = Parallel(n_jobs=num_cores)(delayed(generateData)
                            (i, data_size, batch_size, l, out_mesh, tmax, out_tmesh, shift, kappa_mean, kappa_std)
                             for i in range(int(data_size/batch_size)))

    u_list, kappa_list = [[data[i] for data in out_list] for i in range(2)]

    u_list = np.concatenate(u_list)
    kappa_list = np.concatenate(kappa_list)

    ## shape of u_list = (data_size, data_channels, out_tmesh, out_mesh)
    print(u_list.shape)
    print(u_list.dtype)

    ## shape of kappa_list = (data_size, 1)
    print(kappa_list.shape)
    print(kappa_list.dtype)

    print('Exporting to: ' + FILENAME)
    np.savez(FILENAME, x=u_list, params=np.stack([kappa_list.flatten()], axis=1))

