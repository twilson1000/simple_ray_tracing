#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.pyplot as plt
import numpy as np
import pathlib

# Local imports
from simple_ray_tracing.numerics import RK45

logger = logging.getLogger(__name__)

this_directory = pathlib.Path(__file__).parent

class SimpleHarmonicOscillator:
    __slots__ = ("a", "b", "w")

    def __init__(self, a: float, b: float):
        self.a = float(a)
        self.b = float(b)
        self.w = (-self.a * self.b)**0.5

    def ode_rhs(self, t, y):
        return np.array([self.a * y[1], self.b * y[0]])
    
    def exact_solution(self, t, t0, y0):
        t = np.array(t)
        y = np.empty((*t.shape, 2))

        tau = self.w * (t - t0) 
        A, B = y0[1] / self.w, y0[0]

        y[..., 0] = A * np.sin(tau) + B * np.cos(tau)
        y[..., 1] = A * self.w * np.cos(tau) - B * self.w * np.sin(tau)

        return y

def test_simple_harmonic_oscillator():
    '''
    Test simple harmonic oscillator
    '''
    sho = SimpleHarmonicOscillator(1.0, -1.0)
    t0, y0 = 0.0, np.array([0.0, 1.0])

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position [m]")
    ax1.grid()

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Global Error []")
    ax2.set_yscale('log')
    ax2.grid()

    solver = RK45(t0, y0, sho.ode_rhs, use_adaptive_timestep=True)

    n_steps = 10
    time = np.zeros(n_steps)
    y = np.zeros((n_steps, 2))

    time[0] = t0
    y[0] = y0
    interpolant_coefficients = np.zeros((n_steps, 2, solver.interpolant_order))

    logger.info(f"Initial step = {solver.timestep}")

    for i in range(1, n_steps):
        solver.step_adaptive()
        time[i] = solver.time
        y[i] = solver.state_vector
        interpolant_coefficients[i] = solver.interpolant_coefficients

        if solver.stop_condition:
            break

    ax1.plot(time, y[:, 0], color="red", label="RK45", marker="o")

    y_exact = sho.exact_solution(time, t0, y0)
    
    error = abs(y[:, 0] - y_exact[:, 0])
    ax2.plot(time, error, color="red", label="RK45", marker="o")

    tt = np.linspace(t0, time[-1], 101)
    y_exact = sho.exact_solution(tt, t0, y0)
    ax1.plot(tt, y_exact[:, 0], color='black', label="Exact", ls="dotted")

    ax1.legend(loc="upper left")
    fig1.tight_layout()
    
    ax2.legend(loc="upper left")
    fig2.tight_layout()

    fig1.savefig(this_directory.joinpath("simple-harmonic-oscillator_solution.png"))
    fig2.savefig(this_directory.joinpath("simple-harmonic-oscillator_error.png"))
    plt.show()

def main():
    test_simple_harmonic_oscillator()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H-%M-%S"
    )
    main()