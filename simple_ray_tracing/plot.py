#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

# Local imports

logger = logging.getLogger(__name__)

def plot_ray(output_file, ray_index: int, savename: str=None):
    '''
    Plot a ray.
    '''
    fig, axes = plt.subplots(figsize=(10, 8), ncols=4, nrows=2)

    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    
    ax_NxNy = axes[0, 1]
    ax_NxNy.set_xlabel("Time [ns]")
    ax_NxNy.set_ylabel("N []")

    ax_X = axes[0, 2]
    ax_X.set_xlabel("Time [ns]")
    ax_X.set_ylabel("Normalised density X []")

    ax_Z = axes[0, 3]
    ax_Z.set_xlabel("Time [ns]")
    ax_Z.set_ylabel("Normalised damping frequency Z []")

    ax_phase = axes[1, 0]
    ax_phase.set_xlabel("Time [ns]")
    ax_phase.set_ylabel(r"Phase / 2$\pi$ []")

    ax_H = axes[1, 1]
    ax_H.set_xlabel("Time [ns]")
    ax_H.set_ylabel("Hamiltonian []")
    
    ax_tau = axes[1, 2]
    ax_tau.set_xlabel("Time [ns]")
    ax_tau.set_ylabel(r"Optical depth $\tau$ []")

    ax_P = axes[1, 3]
    ax_P.set_xlabel("Time [ns]")
    ax_P.set_ylabel("Normalised Power []")

    with netCDF4.Dataset(output_file, "r") as dset:
        ray_dset = dset[f'ray_{ray_index}']
        time = ray_dset['time'][:]
        position = ray_dset['position'][:]
        refractive_index = ray_dset['refractive_index'][:]
        phase = ray_dset['phase'][:]
        hamiltonian = ray_dset['hamiltonian'][:, 0]
        normalised_density = ray_dset['normalised_density'][:]
        normalised_damping_frequency = ray_dset['normalised_damping_frequency'][:]
        optical_depth = ray_dset['optical_depth'][:]
    
    ax_xy.plot(position[:, 0], position[:, 1], color="black")
    ax_NxNy.plot(time, refractive_index[:, 0], color="black", label=r"$N_x$")
    ax_NxNy.plot(time, refractive_index[:, 1], color="red", label=r"$N_y$")
    ax_NxNy.legend(loc="upper left")
    ax_X.plot(time, normalised_density, color="black")
    ax_Z.plot(time, normalised_damping_frequency, color="black")
    ax_phase.plot(time, phase / (2 * np.pi), color="black")
    ax_H.plot(time, hamiltonian, color="black")
    ax_tau.plot(time, optical_depth, color="black")
    ax_P.plot(time, np.exp(-optical_depth), color="black")

    fig.tight_layout()
    if savename is not None:
        fig.savefig(savename)
    plt.show()
