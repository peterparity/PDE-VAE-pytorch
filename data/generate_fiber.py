#!/usr/bin/env python

"""
generate_fiber.py

Script for generating the nonlinear fiber propagation dataset.
Requires MEEP (https://meep.readthedocs.io/en/latest/).
"""

import shutil

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob

from joblib import Parallel, delayed
import multiprocessing

import warnings
import argparse

def get_f(total_time, dT, fsrc, batch_size):
    mesh = int(total_time/dT)
    assert mesh == total_time/dT
    T = 1/fsrc
    n_periods = 20
    
    k = 2*np.pi * np.fft.fftfreq(mesh, d=dT)
    w = np.pi/(n_periods*T)
    envelope = np.exp(-k**2/(2*w**2))
    v0 = envelope * (np.random.normal(size=(batch_size, mesh)) + 1j*np.random.normal(size=(batch_size, mesh)))
    u0 = np.fft.ifft(v0)
    u0 = np.sqrt(2 * mesh) * u0/np.expand_dims(np.linalg.norm(u0, axis=-1), axis=-1) # normalize
    v0 = np.fft.fft(u0)
    
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    return u0, lambda t: 1/mesh * np.matmul(v0, np.exp(1j * k * t)) * sigmoid(t/(20*T)-5) * np.exp(-1j*2*np.pi*fsrc*t) # slowly turn on

def get_f_half(total_time, dT, fsrc, batch_size):
    T = 1/fsrc
    u0, f = get_f(total_time, dT, fsrc, batch_size)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    return u0, lambda t: f(t) * sigmoid((total_time/2-t)/(20*T)-5) # turn off before half total_time

def generateData(i, resolution=5, total_time=10000, dT=10, length=500., pml_thickness=10., outer_radius=10., chi3=0.0, half=False, shift=200):
    ### Generate data using MEEP

    print('Run ' + str(i+1) + ' of ' + str(data_size))

    while True:
        dr1 = np.random.normal(scale=0.05)
        dr2 = np.random.normal(scale=0.02)
        de1 = np.random.normal(scale=1)
        de2 = np.random.normal(scale=2)

        inner_core_radius = 0.5 + dr2
        core_radius = 1.0 + dr1

        if core_radius > inner_core_radius:
            break


    output_dir = f"out-{i:05d}"

    cell_size = mp.Vector3(outer_radius + pml_thickness, 0, length + 2*pml_thickness)
    pml_layers = [mp.PML(thickness=pml_thickness)]
    default_material = mp.Medium(index=1, chi3=chi3)
    geometry = [mp.Block(center=mp.Vector3(), size=mp.Vector3(2*core_radius, mp.inf, mp.inf), material=mp.Medium(epsilon=8 + de1, chi3=chi3)),
                mp.Block(center=mp.Vector3(), size=mp.Vector3(2*inner_core_radius, mp.inf, mp.inf), material=mp.Medium(epsilon=30 + de2, chi3=chi3))
               ]

    fsrc = 0.1
    if not half:
        u0, f = get_f(total_time, dT, fsrc=fsrc, batch_size=1)
    else:
        u0, f = get_f_half(total_time, dT, fsrc=fsrc, batch_size=1)
    sources = [mp.Source(src=mp.CustomSource(src_func=lambda t: f(t)[0]),
                             center=mp.Vector3(0,0,-length/2.),
                             size=mp.Vector3(2*(3)),
                             component=mp.Er)]

    sim = mp.Simulation(cell_size=cell_size,
                        resolution=resolution,
                        boundary_layers=pml_layers,
                        sources=sources,
                        geometry=geometry,
                        dimensions=mp.CYLINDRICAL,
                        m=1
                       )
    flux_total = sim.add_flux(fsrc, 1.*fsrc, int(fsrc*total_time)+1, mp.FluxRegion(center=mp.Vector3(0, 0, -length/2. + pml_thickness), size=mp.Vector3(2*outer_radius)))

    sim.use_output_directory(output_dir)
    sim.run(mp.at_every(dT, mp.in_volume(mp.Volume(center=mp.Vector3(), size=mp.Vector3(0,0,length)), mp.output_efield_r)), until=total_time)

    files = sorted(glob.glob(output_dir + "/*er-*.h5"))
    data = []
    for file in files:
        f = h5py.File(file, "r")
        data.append(np.array(f['er.r']) + 1j * np.array(f['er.i']))
    
    data = np.stack(data)

    # Normalize by flux
    freqs = np.array(mp.get_flux_freqs(flux_total))
    flux = np.array(mp.get_fluxes(flux_total))
    integrated_flux = np.sum(flux)*(freqs[1]-freqs[0])
    integrated_efield = np.sum(np.abs(data[:, int(resolution*pml_thickness)+1])**2)*dT
    norm_factor = np.sqrt(integrated_flux/integrated_efield)
    data *= norm_factor

    mean_norm2 = np.mean(np.abs(data[:, int(resolution*pml_thickness)+1])**2)

    # Remove carrier frequency/wavelength
    k = np.fft.fftfreq(data.shape[-1], d=1./resolution)
    k0 = k[np.argmax(np.abs(np.mean(np.fft.fft(data), 0)))]
    psi = data * np.exp(1j * 2*np.pi * (fsrc * dT * np.expand_dims(np.arange(data.shape[0]),1) - k0 * np.arange(data.shape[1])/resolution))

    psi = psi[shift:, int(resolution*pml_thickness)+1:-1:int(resolution)] # drop region near PML and initial time 200*dT
    psi = psi.transpose() # swap t and z axis

    shutil.rmtree(output_dir)

    return np.expand_dims(np.stack([np.real(psi), np.imag(psi)], axis=-3), 0).astype(np.float32), np.array([[dr1, dr2, de1, de2, k0, mean_norm2]]).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fiber dataset using MEEP.")
    parser.add_argument('-d', '--datasize', type=int, default=200, 
        help="Size of generated dataset--should be a multiple of batch_size. Default: 200")
    parser.add_argument('-f', '--filename', type=str, default='fiber_dataset.npz', 
        help="Path with file name ending in .npz where dataset is saved. Default: fiber_dataset.npz")
    parser.add_argument('--half', action='store_true', 
        help="Stop input pulse before half of the total simulation time to ensure entire pulse passes during simulation.")

    args = parser.parse_args()
    data_size = args.datasize
    FILENAME = args.filename

    length = 510
    total_time = 10000
    chi3 = 0.02
    shift = 0 if args.half else 200

    num_cores = multiprocessing.cpu_count() # may want to use less than maximum number of CPUs
    print('Using ' + str(num_cores) + ' cores...')
    out_list = Parallel(n_jobs=num_cores)(
        delayed(generateData)(i, total_time=total_time, length=length, chi3=chi3, half=args.half, shift=shift) for i in range(data_size))

    u_list, params_list = [[data[i] for data in out_list] for i in range(2)]

    u_list = np.concatenate(u_list)
    params_list = np.concatenate(params_list)

    # Re-normalize to achieve component variance ~ 1 on average over the dataset
    # norm_factor = np.sqrt(2/np.mean(params_list[:,-1]))
    norm_factor = 0.8600365727096997
    u_list *= norm_factor

    ## shape of u_list = (data_size, data_channels, length-10, total_time/dT - shift)
    print(u_list.shape)
    print(u_list.dtype)

    ## shape of params_list = (data_size, 6)
    print(params_list.shape)
    print(params_list.dtype)

    print('Exporting to: ' + FILENAME)
    np.savez(FILENAME, x=u_list, params=params_list)








