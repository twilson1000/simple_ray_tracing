#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pathlib

# Local imports
from simple_ray_tracing import (UnmagnetisedColdPlasma, ConstantValueModel,
    RayOptions, SimpleRayTracer, plot_ray)

logger = logging.getLogger(__name__)

this_directory = pathlib.Path(__file__).parent

def vacuum_ray_test():
    density_model = ConstantValueModel(0.0)
    damping_frequency_model = ConstantValueModel(0.0)
    hamiltonian_model = UnmagnetisedColdPlasma()
    options = RayOptions(domain=[[-1, 1], [-1, 1], [-1, 1]])

    output_filepath = this_directory.joinpath("vacuum_ray.nc")

    ray_tracer = SimpleRayTracer(density_model, damping_frequency_model,
        hamiltonian_model, options, output_filepath)
    
    angle = np.pi / 4
    ray_tracer.trace_ray(0.0, np.zeros(3),
        np.array([np.cos(angle), np.sin(angle), 0.0]), 28.0, 0.0)

    plot_ray(output_filepath, 0, this_directory.joinpath("vacuum_ray.png"))

def main():
    vacuum_ray_test()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H-%M-%S"
    )
    main()