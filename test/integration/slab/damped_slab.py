#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pathlib

# Local imports
from simple_ray_tracing import (UnmagnetisedColdPlasma, ConstantValueModel,
    C2Ramp, NormalisedValueModel, RayOptions, SimpleRayTracer, plot_ray)

logger = logging.getLogger(__name__)

this_directory = pathlib.Path(__file__).parent

def undamped_slab():
    frequency_ghz = 28.0

    x0, X0, X1, LX_inverse = 0.0, 0.0, 2.0, 0.5
    X_ramp = C2Ramp(x0, X0, X1, LX_inverse)
    X_to_ne = 1e18 * (frequency_ghz / 9)**2
    density_model = NormalisedValueModel(X_ramp, X_to_ne)

    damping_frequency_model = ConstantValueModel(0.0)

    hamiltonian_model = UnmagnetisedColdPlasma()
    options = RayOptions(domain=[[-1, 5], [-2, 2], [-1, 1]],
        use_adaptive_timestep=True)

    output_filepath = this_directory.joinpath("undamped_slab.nc")

    ray_tracer = SimpleRayTracer(density_model, damping_frequency_model,
        hamiltonian_model, options, output_filepath)
    
    angle = np.pi / 8
    ray_tracer.trace_ray(0.0, np.array([-1, 0, 0]),
        np.array([np.cos(angle), np.sin(angle), 0.0]), frequency_ghz, 0.0)

    plot_ray(output_filepath, 0, this_directory.joinpath("undamped_slab.png"))

def damped_slab():
    frequency_ghz = 28.0

    x0, X0, X1, LX_inverse = 0.0, 0.0, 2.0, 0.5
    X_ramp = C2Ramp(x0, X0, X1, LX_inverse)
    X_to_ne = 1e18 * (frequency_ghz / 9)**2
    density_model = NormalisedValueModel(X_ramp, X_to_ne)

    x0, Z0, Z1, LZ_inverse = 0.0, 0.0, 1e-4, 0.5
    Z_ramp = C2Ramp(x0, Z0, Z1, LZ_inverse)
    Z_to_nu = frequency_ghz
    damping_frequency_model = NormalisedValueModel(Z_ramp, Z_to_nu)

    hamiltonian_model = UnmagnetisedColdPlasma()
    options = RayOptions(domain=[[-1, 5], [-2, 2], [-1, 1]],
        use_adaptive_timestep=True)

    output_filepath = this_directory.joinpath("damped_slab.nc")

    ray_tracer = SimpleRayTracer(density_model, damping_frequency_model,
        hamiltonian_model, options, output_filepath)
    
    angle = np.pi / 8
    ray_tracer.trace_ray(0.0, np.array([-1, 0, 0]),
        np.array([np.cos(angle), np.sin(angle), 0.0]), frequency_ghz, 0.0)

    plot_ray(output_filepath, 0, this_directory.joinpath("damped_slab.png"))


def main():
    undamped_slab()
    damped_slab()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H-%M-%S"
    )
    main()