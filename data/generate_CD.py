#!/usr/bin/env python

"""
generate_CD.py

Script for generating the 2D convection-diffusion dataset.
"""

import os
import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.stats import truncnorm

import argparse


parser = argparse.ArgumentParser(description="Generate convection-diffusion dataset.")
parser.add_argument('-d', '--datasize', type=int, default=1000, 
    help="Size of generated dataset--should be a multiple of batch_size. Default: 1000")
parser.add_argument('-b', '--batchsize', type=int, default=50, 
    help="Batch size for generating dataset in parallel--should divide data_size. Default: 50")
parser.add_argument('-f', '--filename', type=str, default='CD_dataset.npz', 
    help="Path with file name ending in .npz where dataset is saved. Default: CD_dataset.npz")

args = parser.parse_args()
data_size = args.datasize
batch_size = args.batchsize
FILENAME = args.filename

l = 16*np.pi # system size
mesh = 256 # mesh   
tmax = 4*np.pi # max time
tmesh = 64
dt = tmax/tmesh # time step
shift = 0 # shift time to exclude initial conditions, set to 0 to keep t = 0
 
dmean = 0.1 # diffusion constant
dstd = dmean/4
velstd = 0.5 # standard deviation of velocity

kx = np.expand_dims(2*np.pi * np.fft.fftfreq(mesh, d=l/mesh), axis=-1)
ky = np.expand_dims(2*np.pi * np.fft.fftfreq(mesh, d=l/mesh), axis=0)

# for use in 1st derivative
kx_1 = kx.copy()
kx_1[int(mesh/2)] = 0
ky_1 = ky.copy()
ky_1[:, int(mesh/2)] = 0


### Generate data
u_list = []
d_list = []
velx_list = []
vely_list = []
for i in range(int(data_size/batch_size)):
    print('Batch ' + str(i+1) + ' of ' + str(int(data_size/batch_size)))

    d = truncnorm.rvs((0 - dmean) / dstd, (2 * dmean - dmean) / dstd, 
                        loc=dmean, scale=dstd, size=(batch_size, 1, 1, 1))
    d_list.append(d.astype(np.float32))

    velx = np.random.normal(loc=0, scale=velstd, size=(batch_size, 1, 1, 1))
    vely = np.random.normal(loc=0, scale=velstd, size=(batch_size, 1, 1, 1))
    velx_list.append(velx.astype(np.float32))
    vely_list.append(vely.astype(np.float32))

    ## initial condition
    krange = (0.25 * mesh*np.pi/l - 8 * np.pi/l) * np.random.rand(batch_size, 1, 1, 1) + 8 * np.pi/l
    envelope = np.exp(-1/(2*krange**2) * (kx**2 + ky**2) )
    v0 = envelope * (np.random.normal(loc=0, scale=1.0, size=(batch_size, 1, mesh, mesh)) 
                        + 1j*np.random.normal(loc=0, scale=1.0, size=(batch_size, 1, mesh, mesh)))
    u0 = np.real(ifft2(v0))
    u0 = mesh * u0/np.linalg.norm(u0, axis=(-2,-1), keepdims=True) # normalize
    v0 = fft2(u0)

    ## Differential equation
    L = -d * (kx**2 + ky**2) - 1j * (kx_1 * velx + ky_1 * vely)

    t = np.linspace(shift, tmax + shift, tmesh, endpoint=False)
    v = np.exp(np.expand_dims(np.expand_dims(t, -1), -1) * L) * v0
    u = np.real(ifft2(v))
    u_list.append(np.expand_dims(u, axis=1).astype(np.float32))


u_list = np.concatenate(u_list)
d_list = np.concatenate(d_list)
velx_list = np.concatenate(velx_list)
vely_list = np.concatenate(vely_list)

## shape of u_list = (data_size, 1, tmesh, mesh, mesh)
print(u_list.shape)
print(u_list.dtype)

## shape of d_list = (data_size, 1, 1, 1)
print(d_list.shape)
print(d_list.dtype)

print('Exporting to: ' + FILENAME)
np.savez(FILENAME, x=u_list, 
    params=np.stack([d_list.flatten(), velx_list.flatten(), vely_list.flatten()], axis=1))
