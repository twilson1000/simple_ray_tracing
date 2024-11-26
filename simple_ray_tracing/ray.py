#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.constants as const

# Local imports
from .hamiltonian import HamiltonianModel
from .numerics import RK45
from .plasma import ValueModel, ValueDerivativeModel

logger = logging.getLogger(__name__)

class RayTrajectory:
    __slots__ = ("dimension", "cache_size", "index", "final_index", 
        "time_ns", "position_m", "refractive_index", "frequency_ghz",
        "vacuum_wavenumber_per_m", "critical_density_per_m3", "phase",
        "density_per_m3", "normalised_density", "damping_frequency",
        "normalised_damping_frequency", "hamiltonian", "ray_velocity_x",
        "ray_velocity_N", "damping_rate_collisional", "optical_depth",
        "dispersion_tensor", "interpolant_coefficients")
    
    def __init__(self, dimension: int, cache_size: int, interpolant_shape: int):
        '''
        
        '''
        self.dimension = dimension
        self.cache_size = cache_size

        self.index = 0
        self.final_index = 0

        # Constant values.
        self.frequency_ghz = 0.0
        self.vacuum_wavenumber_per_m = 0.0
        self.critical_density_per_m3 = 0.0

        # Values along the ray trajectory.
        shape_0d = (self.cache_size,)
        shape_1d = (self.cache_size, dimension)
        shape_2d = (self.cache_size, dimension, dimension)

        self.time_ns = np.zeros(shape_0d)
        self.position_m = np.zeros(shape_1d)
        self.refractive_index = np.zeros(shape_1d)
        self.phase = np.zeros(shape_0d)
        self.density_per_m3 = np.zeros(shape_0d)
        self.normalised_density = np.zeros(shape_0d)
        self.damping_frequency = np.zeros(shape_0d)
        self.normalised_damping_frequency = np.zeros(shape_0d)
        self.hamiltonian = np.zeros(shape_0d, dtype=complex)
        self.ray_velocity_x = np.zeros(shape_1d)
        self.ray_velocity_N = np.zeros(shape_1d)
        self.damping_rate_collisional = np.zeros(shape_0d)
        self.optical_depth = np.zeros(shape_0d)
        self.dispersion_tensor = np.zeros(shape_2d, dtype=complex)
        self.interpolant_coefficients = np.zeros(
            (self.cache_size, *interpolant_shape))
    
    def write_constants(self, ray: "Ray"):
        '''
        
        '''
        self.frequency_ghz = ray.frequency_ghz
        self.vacuum_wavenumber_per_m3 = ray.vacuum_wavenumber_per_m
        self.critical_density_per_m3 = ray.critical_density_per_m3

    def write_step(self, index: int, ray: "Ray"):
        '''
        
        '''
        self.position_m[index] = ray.position_m
        self.refractive_index[index] = ray.refractive_index
        self.time_ns[index] = ray.time_ns
        self.phase[index] = ray.phase
        self.density_per_m3[index] = ray.density_per_m3
        self.normalised_density[index] = ray.normalised_density
        self.damping_frequency[index] = ray.damping_frequency
        self.normalised_damping_frequency[index] = \
            ray.normalised_damping_frequency
        self.hamiltonian[index] = ray.hamiltonian
        self.ray_velocity_x[index] = ray.ray_velocity_x
        self.ray_velocity_N[index] = ray.ray_velocity_N
        self.damping_rate_collisional[index] = ray.damping_rate_collisional
        self.optical_depth[index] = ray.optical_depth
        self.dispersion_tensor[index] = ray.dispersion_tensor
        self.interpolant_coefficients[index] = ray.interpolant_coefficients

    def finalise(self, final_index: int):
        '''

        '''
        self.final_index = final_index

class RayOptions:
    __slots__ = ("dimension", "max_elements", "min_step", "max_step",
        "absolute_tolerance", "relative_tolerance", "max_iterations",
        "use_adaptive_timestep", "domain", "max_optical_depth")

    def __init__(self, dimension: int=3, max_elements: int=100,
        min_step: float=1e-8, max_step: float=np.inf,
        absolute_tolerance: float=1e-6, relative_tolerance: float=1e-4,
        max_iterations: int=64, use_adaptive_timestep: bool=True,
        domain: npt.NDArray[float]=None, max_optical_depth: float=100) -> None:
        '''
        
        '''
        self.dimension = int(dimension)
        self.max_elements = int(max_elements)
        self.min_step = float(min_step)
        self.max_step = float(max_step)
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.max_iterations = int(max_iterations)
        self.use_adaptive_timestep = use_adaptive_timestep

        if domain is None:
            self.domain = np.array(
                [[-np.inf, np.inf] for _ in range(self.dimension)])
        else:
            self.domain = np.array(domain, dtype=float)
        
        self.max_optical_depth = float(max_optical_depth)

class Ray:
    __slots__ = ("density_model", "damping_frequency_model",
        "hamiltonian_model", "options", "trajectory", "integrator",
        "time_ns", "position_m", "refractive_index", "frequency_ghz",
        "vacuum_wavenumber_per_m", "phase", "refractive_index_squared",
        "density_per_m3", "critical_density_per_m3", "normalised_density",
        "normalised_density_first_derivative",
        "normalised_density_second_derivative", "damping_frequency",
        "normalised_damping_frequency", "hamiltonian", "ray_velocity_x",
        "ray_velocity_N", "damping_rate_collisional", "optical_depth",
        "dispersion_tensor", "interpolant_coefficients",
        )

    speed_of_light_m_per_ns = 1e-9 * const.speed_of_light

    def __init__(self, density_model: ValueDerivativeModel,
        damping_frequency_model: ValueModel,
        hamiltonian_model: HamiltonianModel, options: RayOptions):
        '''
        
        '''
        self.density_model = density_model
        self.damping_frequency_model = damping_frequency_model
        self.hamiltonian_model = hamiltonian_model
        self.options = options

        n = self.options.dimension
        
        self.position_m = np.zeros(n)
        self.refractive_index = np.zeros(n)
        self.refractive_index_squared = np.zeros(())
        self.time_ns = np.zeros(())
        self.frequency_ghz = np.zeros(())
        self.vacuum_wavenumber_per_m = np.zeros(())
        self.phase = np.zeros(())

        self.density_per_m3 = np.zeros(())
        self.critical_density_per_m3 = np.zeros(())
        self.normalised_density = np.zeros(())
        self.normalised_density_first_derivative = np.zeros(n)
        self.normalised_density_second_derivative = np.zeros((n, n))
        self.damping_frequency = np.zeros(())
        self.normalised_damping_frequency = np.zeros(())
        self.hamiltonian = np.zeros((), dtype=complex)
        self.ray_velocity_x = np.zeros(n)
        self.ray_velocity_N = np.zeros(n)
        self.damping_rate_collisional = np.zeros(())
        self.optical_depth = np.zeros(())
        self.dispersion_tensor = np.zeros((n, n), dtype=complex)

        self.interpolant_coefficients = np.zeros((2*n, RK45.interpolant_order))

        self.trajectory = RayTrajectory(self.options.dimension,
            self.options.max_elements, (2*n, RK45.interpolant_order))

    @classmethod
    def with_initial_values(cls, density_model: ValueDerivativeModel,
        damping_frequency_model: ValueModel,
        hamiltonian_model: HamiltonianModel, options: RayOptions,
        time_ns, position_m, refractive_index, frequency_ghz, phase):
        '''
        
        '''
        obj = cls(density_model, damping_frequency_model, hamiltonian_model,
            options)
        obj.set_frequency(frequency_ghz)
        obj.position_m[...] = position_m
        obj.refractive_index[...] = refractive_index
        obj.time_ns[...] = time_ns
        obj.phase[...] = phase
        return obj

    def set_frequency(self, frequency_ghz):
        '''
        
        '''
        self.frequency_ghz[...] = frequency_ghz
        self.critical_density_per_m3[...] = 1e18 * (frequency_ghz / 9)**2
        self.vacuum_wavenumber_per_m[...] = frequency_ghz / self.speed_of_light_m_per_ns

    def set_position(self, time_ns, position_m, refractive_index):
        '''
        Set all ray variables to values at current position.
        '''
        # Phase space position.
        self.position_m[...] = position_m
        self.refractive_index[...] = refractive_index
        self.time_ns[...] = time_ns

        # Refractive index.
        self.refractive_index_squared = np.dot(self.refractive_index,
            self.refractive_index)

        # Density.
        self.density_per_m3[...] = self.density_model.value(self.position_m)
        self.normalised_density = self.density_per_m3 / self.critical_density_per_m3
        self.normalised_density_first_derivative[...] = \
            (self.density_model.value_first_derivative(self.position_m)
                / self.critical_density_per_m3)
        self.normalised_density_second_derivative[...] = \
            (self.density_model.value_second_derivative(self.position_m)
                / self.critical_density_per_m3)

        # Collisional damping.
        self.damping_frequency[...] = self.damping_frequency_model.value(
            self.position_m)
        self.normalised_damping_frequency[...] = (self.damping_frequency
            / self.frequency_ghz)

        # Hamiltonian.
        self.hamiltonian[...] = self.hamiltonian_model.hamiltonian(
            self.normalised_density, self.refractive_index_squared,
            self.normalised_damping_frequency)
        
        # Velocities.
        c = self.speed_of_light_m_per_ns
        self.ray_velocity_x[...] = (c
            * self.hamiltonian_model.normalised_ray_velocity_x(
                refractive_index))
        self.ray_velocity_N[...] = (c
            * self.hamiltonian_model.normalised_ray_velocity_N(
                self.normalised_density,
                self.normalised_density_first_derivative,
                self.refractive_index_squared))
        
        # Damping rates.
        self.damping_rate_collisional = \
            self.hamiltonian_model.damping_rate_collisional(
                self.hamiltonian.imag, self.normalised_density,
                self.refractive_index_squared)

        # Dispersion tensor.
        self.dispersion_tensor[...] = self.hamiltonian_model.dispersion_tensor(
            self.normalised_density, self.refractive_index_squared)

    def get_state_vector_derivative(self, time, state_vector):
        '''
        Get derivative dy/dt of state vector y = (x, N, phi, tau):
            x = position
            N = refractive index
            phi = phase
            tau = optical depth
        '''
        n = self.options.dimension
        self.set_position(time, state_vector[:n], state_vector[n: 2 * n])

        state_vector_derivative = np.zeros(2*n + 2)
        state_vector_derivative[:n] = self.ray_velocity_x
        state_vector_derivative[n: 2 * n] = self.ray_velocity_N

        # Wavevector is spatial gradient of phase.
        state_vector_derivative[2 * n] = (self.vacuum_wavenumber_per_m
            * np.dot(self.refractive_index, self.ray_velocity_x))

        # Damping rate is the rate of change of optical depth.
        state_vector_derivative[2 * n + 1] = self.damping_rate_collisional 

        return state_vector_derivative

    def finalise_step(self, time, state_vector):
        '''
        
        '''
        n = self.options.dimension
        self.set_position(time, state_vector[:n], state_vector[n: 2 * n])
        self.phase[...] = state_vector[2 * n]
        self.optical_depth[...] = state_vector[2 * n + 1]

    def stop_out_of_domain(self, x, xmin, xmax):
        msg = f"Ray out of domain ({xmin:5.3e}, {xmax:5.3e}): {x:5.3e}"
        logger.warning(msg)
        return msg

    def stop_reached_max_optical_depth(self, optical_depth, max_optical_depth):
        msg = ("Reached max optical depth: "
            f"{optical_depth:5.3e} >= {max_optical_depth:5.3e}")
        logger.warning(msg)
        return msg

    def stop_reached_max_elements(self, max_ray_elements):
        msg = f"Reached max ray elements: {max_ray_elements}"
        logger.error(msg)
        return msg

    def trace(self):
        n = self.position_m.size
        initial_state_vector = np.zeros(2*n + 2)
        initial_state_vector[:n] = self.position_m
        initial_state_vector[n: 2 * n] = self.refractive_index

        self.integrator = RK45(self.time_ns, initial_state_vector,
            self.get_state_vector_derivative, self.options.min_step,
            self.options.max_step, self.options.absolute_tolerance,
            self.options.relative_tolerance, self.options.max_iterations,
            self.options.use_adaptive_timestep)
        
        # Write initial conditions to trajectory.
        self.trajectory.write_step(0, self)

        # Start integration loop.
        for element_index in range(1, self.options.max_elements):
            # Take integration step.
            self.integrator.step()
            self.finalise_step(self.integrator.time, self.integrator.state_vector)
            self.trajectory.write_step(element_index, self)

            # Check stop condition on integrator.
            if self.integrator.stop_condition:
                break

            # Check stop condition on domain.
            for i in range(n):
                x, (xmin, xmax) = self.position_m[i], self.options.domain[i]

                if (x < xmin) or (x > xmax):
                    self.integrator.stop_condition = self.stop_out_of_domain(
                        x, xmin, xmax)
                    break
            
            if self.integrator.stop_condition:
                break

            # Check stop condition on optical depth.
            if self.optical_depth > self.options.max_optical_depth:
                self.integrator.stop_condition = \
                    self.stop_reached_max_optical_depth(self.optical_depth,
                        self.options.max_optical_depth)
            
            if self.integrator.stop_condition:
                break
        
        # End of integration loop, finalise trajectory.
        self.trajectory.finalise(element_index)

        # Check if we exceeded maximum ray elements.
        if element_index == self.options.max_elements - 1:
            self.integrator.stop_condition = \
                self.stop_reached_max_elements(self.options.max_elements)

