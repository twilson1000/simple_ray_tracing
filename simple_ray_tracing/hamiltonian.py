#!/usr/bin/python3

# Standard imports
import abc
import enum
import logging
import numpy as np
import numpy.typing as npt

# Local imports

logger = logging.getLogger(__name__)

class HamiltonianModel(abc.ABC):
    '''  '''

    @abc.abstractmethod
    def hamiltonian(self) -> complex:
        ''' Ray tracing hamiltonian '''

    @abc.abstractmethod
    def dispersion_tensor(self) -> npt.NDArray[complex]:
        ''' Dispersion tensor. '''

    @abc.abstractmethod
    def normalised_ray_velocity_x(self) -> npt.NDArray[float]:
        ''' Ray spatial velocity normalised to speed of light. '''
    
    @abc.abstractmethod
    def normalised_ray_velocity_N(self) -> npt.NDArray[float]:
        ''' Ray refractive index velocity normalised to speed of light. '''
    
    @abc.abstractmethod
    def damping_rate_collisional(self) -> float:
        ''' Damping rate due to collisions. '''

class UnmagnetisedColdPlasma:
    '''  '''
    __slots__ = ()

    def hamiltonian(self, X, N2, Z) -> complex:
        '''
        '''
        U = 1 - 1.0j * Z
        return U - X - N2

    def dispersion_tensor(self, X, N2):
        '''
        '''
        D = np.zeros((3, 3), dtype=complex)
        D[0, 0] = 1 - X - N2
        D[1, 1] = D[0, 0]
        D[2, 2] = 1 - X

        return D

    def normalised_ray_velocity_x(self, N):
        '''
        '''
        return N
    
    def normalised_ray_velocity_N(self, X, dX_dx, N2):
        '''
        '''
        return -dX_dx / (2 * (X + N2))

    def damping_rate_collisional(self, H_imag, X, N2):
        return abs(H_imag / (X + N2))
    