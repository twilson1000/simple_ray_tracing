#!/usr/bin/python3

# Standard imports
import datetime
import logging
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pathlib

# Local imports
from .ray import Ray, RayOptions
from . import __author__, __version__, __url__

logger = logging.getLogger(__name__)

class OutputFile:
    __slots__ = ("filepath", "dset", "initialised")

    compression_args = {'zlib': True, 'complevel': 7}

    def __init__(self, filepath: str, options: RayOptions) -> None:
        self.filepath = pathlib.Path(filepath)
        self.dset = None

        with netCDF4.Dataset(self.filepath, "w") as dset:
            self.dset = dset
            self.write_input(options)

        self.dset = None

    def __enter__(self):
        self.dset = netCDF4.Dataset(self.filepath, "a")
        return self.dset

    def __exit__(self, *args, **kwargs):
        self.dset.close()
        self.dset = None

    def write_input(self, options: RayOptions):
        '''
        
        '''
        # Write global dimensions.
        self.dset.createDimension("min_max", 2)
        self.dset.createDimension("complex", 2)
        self.dset.createDimension("dimension", options.dimension)

        self.dset.setncattr("author", __author__)
        self.dset.setncattr("creation_time", datetime.datetime.now(
            datetime.timezone.utc).replace(microsecond=0).isoformat(' '))
        self.dset.setncattr("version", __version__)

        # Write inputs.
        input_group = self.dset.createGroup("input")

        scalar_signals = (
            # Name, value, description, unit.
            ("max_elements", options.max_elements, "Max elements per ray", ""),
            ("min_step", options.min_step, "Minimum integration timestep",
                "ns"),
            ("max_step", options.max_step, "Maximum integration timestep",
                "ns"),
            ("absolute_tolerance", options.absolute_tolerance,
                "Absolute error tolerance of integration", ""),
            ("relative_tolerance", options.relative_tolerance,
                "Relative error tolerance of integration", ""),
            ("max_iterations", options.relative_tolerance,
                "Maximum attempts to find adaptive step size", ""),
            ("use_adaptive_timestep", int(options.use_adaptive_timestep),
                "Flag if to use adaptive step size", ""),
            ("max_optical_depth", options.max_optical_depth,
                "Maximum optical depth for a ray", ""),
        )

        for name, value, description, unit in scalar_signals:
            group = input_group.createGroup(name)
            group.value = value
            group.description = description
            group.unit = unit

        array_signals = (
            # Name, value, dimensions, dtype, description, unit.
            ("domain", options.domain, ("dimension", "min_max"), "f4",
                "Min and max value of position in each dimension", "m"),
        )

        for name, value, dimensions, dtype, description, unit in array_signals:
            var = input_group.createVariable(name, dtype, dimensions)
            var[:] = value
            var.description = description
            var.unit = unit

    def write_ray_trajectory(self, ray_name: str, ray: Ray):
        '''
        '''
        assert self.dset is not None, "dset not open"

        ray_group = self.dset.createGroup(ray_name)
        ray_group.stop_condition = ray.integrator.stop_condition

        # Create dimension for ray element.
        n = ray.trajectory.final_index + 1
        state_vector_size = ray.trajectory.interpolant_coefficients.shape[-2]
        interpolant_order = ray.trajectory.interpolant_coefficients.shape[-1]

        ray_group.createDimension("ray_element", n)
        ray_group.createDimension("state_vector", state_vector_size)
        ray_group.createDimension("interpolant_order", interpolant_order)

        dims_0d = ("ray_element",)
        dims_0d_complex = ("ray_element", "complex")
        dims_1d = ("ray_element", "dimension",)
        dims_1d_complex = ("ray_element", "dimension", "complex")
        dims_2d = ("ray_element", "dimension", "dimension")
        dims_2d_complex = ("ray_element", "dimension", "dimension", "complex")

        t = ray.trajectory

        array_signals = (
            # Name, value, dimensions, dtype, description, unit.
            ("time", t.time_ns[:n], dims_0d, "f4", "Time", "ns"),
            ("position", t.position_m[:n], dims_1d, "f4",
                "Spatial position (Cartesian)", "m"),
            ("refractive_index", t.refractive_index[:n], dims_1d, "f4",
                "Refractive index (Cartesian)", ""),
            ("phase", t.phase[:n], dims_0d, "f4", "Eikonal phase (relative)", ""),
            ("density", t.density_per_m3[:n], dims_0d, "f4",
                "Electron density", "m^-3"),
            ("normalised_density", t.normalised_density[:n], dims_0d, "f4",
                "Normalised density X = (f_pe/f)^2", "m^-3"),
            ("damping_frequency", t.damping_frequency[:n], dims_0d, "f4",
                "Damping frequency nu", "GHz"),
            ("normalised_damping_frequency", t.normalised_damping_frequency[:n],
                dims_0d, "f4", "Normalised damping frequency Z=nu/f", "GHz"),
            ("hamiltonian", t.hamiltonian[:n], dims_0d_complex, "f4",
                "Ray tracing Hamiltonian", ""),
            ("ray_velocity_x", t.ray_velocity_x[:n], dims_1d, "f4",
                "First derivative of position with respect to time", "m/ns"),
            ("ray_velocity_N", t.ray_velocity_N[:n], dims_1d, "f4",
                "First derivative of refractive index with respect to time",
                "ns^-1"),
            ("damping_rate_collisional", t.damping_rate_collisional[:n], dims_0d,
                "f4", "Damping rate.", "ns^-1"),
            ("optical_depth", t.optical_depth[:n], dims_0d, "f4",
                "Ray optical depth relative to start", ""),
            ("dispersion_tensor", t.dispersion_tensor[:n], dims_2d_complex, "f4",
                "Wave dispersion tensor", ""),
            ("interpolant_coefficients", t.interpolant_coefficients[:n], 
                ("ray_element", "state_vector", "interpolant_order"), "f4",
                "Polynomial coefficients for interpolating state vector", ""),
        )   
        
        for name, value, dimensions, dtype, description, unit in array_signals:
            logger.info(f"Writing {name}")
            var = ray_group.createVariable(name, dtype, dimensions,
                **self.compression_args)
            var.description = description
            var.unit = unit

            if dimensions[-1] == "complex":
                var[..., 0] = value.real
                var[..., 1] = value.imag
            else:
                var[:] = value
        