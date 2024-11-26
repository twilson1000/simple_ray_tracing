__author__ = "Thomas Wilson"
__version__ = "0.1"
__url__ = "https://github.com/twilson1000/simple_ray_tracing"

from .hamiltonian import UnmagnetisedColdPlasma
from .plasma import (ValueModel, ValueDerivativeModel, ConstantValueModel,
    C2Ramp, QuadraticChannel, QuadraticWell, NormalisedValueModel)
from .plot import plot_ray
from .ray import RayOptions
from .simple_ray_tracer import SimpleRayTracer
