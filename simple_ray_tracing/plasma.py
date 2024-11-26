#!/usr/bin/python3

# Standard imports
import abc
import logging
import numpy as np
import numpy.typing as npt

# Local imports

logger = logging.getLogger(__name__)

class ValueModel(abc.ABC):
    '''
    Model providing a value as a function of Cartesian position.
    '''
    __slots__ = ()
    
    @abc.abstractmethod
    def value(self, position_cartesian):
        '''
        Return value.
        '''

class ValueDerivativeModel(ValueModel):
    '''
    Model providing a value and first and second spatial derivatives as a
    function of Cartesian position.
    '''
    @abc.abstractmethod
    def value_first_derivative(self, position_cartesian):
        '''
        Return first derivative of value with respect to position..
        '''
    
    @abc.abstractmethod
    def value_second_derivative(self, position_cartesian):
        '''
        Return second derivative of value with respect to position.
        '''

class ConstantValueModel(ValueModel):
    __slots__ = ("constant_value",)

    def __init__(self, constant_value: float):
        self.constant_value = float(constant_value)
    
    def value(self, position_cartesian):
        return self.constant_value
    
    def value_first_derivative(self, position_cartesian):
        n = len(position_cartesian)
        return np.zeros(n)
    
    def value_second_derivative(self, position_cartesian):
        n = len(position_cartesian)
        return np.zeros((n, n))

class C2Ramp(ValueDerivativeModel):
    '''
    Density ramps from y0 to y1 in x direction only such that the density
    profile has C2 smoothness.
    '''
    __slots__ = ("x0", "y0", "dy", "Ln_inverse")

    def __init__(self, x0: float, y0: float, y1: float, Ln_inverse: float):
        '''
        '''
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.dy = float(y1 - y0)
        self.Ln_inverse = float(Ln_inverse)

    def normalise_position(self, position):
        return np.clip((position[0] - self.x0) * self.Ln_inverse, 0, 1)

    def value(self, position):
        x = self.normalise_position(position)
        return self.y0 + self.dy * (10 - 15*x + 6*x**2) * x**3

    def value_first_derivative(self, position):
        x = self.normalise_position(position)
        dy_dx = np.zeros_like(position)
        dy_dx[0] = 30 * self.Ln_inverse * self.dy * x**2 * (1 - x)**2
        return dy_dx

    def value_second_derivative(self, position):
        x = self.normalise_position(position)
        n = len(position)
        d2y_dx2 = np.zeros((n, n))
        d2y_dx2[0, 0] = 60 * self.Ln_inverse**2 * self.dy * x * (1 - x) * (1 - 2*x)
        return d2y_dx2

class QuadraticChannel(ValueDerivativeModel):
    '''
    Density is a quadratic well with the well bottom parallel to the y axis.
    '''
    __slots__ = ("origin", "Ln_inverse")

    def __init__(self, origin: npt.NDArray[float], Ln_inverse: float):
        '''
        
        '''
        self.origin = np.array(origin)
        self.Ln_inverse = float(Ln_inverse)

    def value(self, position):
        dx, dz = position[0] - self.origin[0], position[2] - self.origin[2]
        return self.Ln_inverse**2 * (dx**2 + dz**2)

    def value_first_derivative(self, position):
        dy_dx = np.zeros_like(position)
        dy_dx[0] = 2 * position[0]
        dy_dx[2] = 2 * position[2]

        return self.Ln_inverse**2 * dy_dx

    def value_second_derivative(self, position):
        n = len(position)
        d2y_dx2 = np.zeros((n, n))
        d2y_dx2[0, 0] = 2
        d2y_dx2[2, 2] = 2

        return self.Ln_inverse**2 * d2y_dx2

class QuadraticWell(ValueDerivativeModel):
    '''
    Density is a quadratic well centred at origin.
    '''
    __slots__ = ("origin", "Ln_inverse")

    def __init__(self, origin: npt.NDArray[float], Ln_inverse: float):
        '''
        '''
        self.origin = np.array(origin)
        self.Ln_inverse = float(Ln_inverse)
    
    def value(self, position):
        dx = position - self.origin
        return self.Ln_inverse**2 * sum(dx**2)

    def value_first_derivative(self, position):
        dy_dx = np.zeros_like(position)
        dy_dx[:] = 2 * position

        return self.Ln_inverse**2 * dy_dx

    def value_second_derivative(self, position):
        n = len(position)
        d2y_dx2 = np.zeros((n, n))
        d2y_dx2[0, 0] = 2
        d2y_dx2[1, 1] = 2
        d2y_dx2[2, 2] = 2

        return self.Ln_inverse**2 * d2y_dx2

class NormalisedValueModel(ValueDerivativeModel):
    '''
    Model providing a value using a value model which provides a normalised
    value.
    '''
    __slots__ = ("normalised_value_model", "normalising_factor")

    def __init__(self, normalised_value_model: ValueDerivativeModel,
        normalising_factor: float) -> None:
        '''
        
        '''
        self.normalised_value_model = normalised_value_model
        self.normalising_factor = float(normalising_factor)

    def value(self, position):
        return (self.normalising_factor
            * self.normalised_value_model.value(position))

    def value_first_derivative(self, position):
        return (self.normalising_factor
            * self.normalised_value_model.value_first_derivative(position))

    def value_second_derivative(self, position):
        return (self.normalising_factor
            * self.normalised_value_model.value_second_derivative(position))
