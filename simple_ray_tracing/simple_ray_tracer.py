#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from .hamiltonian import HamiltonianModel
from .output import OutputFile
from .plasma import ValueModel, ValueDerivativeModel
from .ray import Ray, RayOptions

logger = logging.getLogger(__name__)

class SimpleRayTracer:
    __slots__ = ("density_model", "damping_frequency_model",
        "hamiltonian_model", "ray_options", "output_file",
        "ray_counter")
    
    def __init__(self, density_model: ValueDerivativeModel,
        damping_frequency_model: ValueModel,
        hamiltonian_model: HamiltonianModel,
        ray_options: RayOptions, output_filepath: str
        ) -> None:
        '''
        
        '''
        self.density_model = density_model
        self.damping_frequency_model = damping_frequency_model
        self.hamiltonian_model = hamiltonian_model
        self.ray_options = ray_options

        self.output_file = OutputFile(output_filepath, ray_options)
        self.ray_counter = 0

    def get_next_ray_name(self) -> str:
        name = f"ray_{self.ray_counter}"
        self.ray_counter += 1
        return name

    def trace_ray(self, time_ns, position_m, refractive_index,
        frequency_ghz, phase):
        '''
        '''
        ray_name = self.get_next_ray_name()
        ray = Ray.with_initial_values(self.density_model,
            self.damping_frequency_model, self.hamiltonian_model,
            self.ray_options, time_ns, position_m, refractive_index,
            frequency_ghz, phase)
        ray.trace()
        
        with self.output_file:
            self.output_file.write_ray_trajectory(ray_name, ray)

    def trace_rays(self, initial_conditions):
        '''
        
        '''
        with self.output_file:
            for (time_ns, position_m, refractive_index, frequency_ghz, phase
                ) in initial_conditions:

                ray_name = self.get_next_ray_name()
                ray = Ray.with_initial_values(self.density_model,
                    self.damping_frequency_model, self.hamiltonian_model,
                    self.ray_options, time_ns, position_m, refractive_index,
                    frequency_ghz, phase)
                ray.trace()

                self.output_file.write_ray_trajectory(ray_name, ray)
