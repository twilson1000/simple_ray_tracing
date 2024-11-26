#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# Local imports

logger = logging.getLogger(__name__)

finite_difference_1st_derivative_2nd_order_stencil = ((-1, -1/2), (1, 1/2))
finite_difference_2nd_derivative_2nd_order_stencil = ((-1, 1.0), (0, -2.0), (1, 1.0))
finite_difference_mixed_2nd_derivative_2nd_order_stencil = ((1, 1, 0.25), (1, -1, -0.25), (-1, 1, -0.25), (-1, -1, 0.25))

def polynomial_interpolate(time, t0, y0, timestep, interpolant_coefficients):
        '''
        Evaluate polynomial on given interval. 
        
        t0 
            Time at start of step interval.
        y0
            State vector at start of step interval.
        timestep
            Time difference across step interval.
        interpolant_coefficients
            Coefficients for polynomial interpolant in acending order
            of power i.e. a[0]*t + a[1]*t**2 + ...
        '''
        t_norm = (time - t0) / timestep
        interpolant_order = len(interpolant_coefficients)
        t_norm_powers = np.cumprod(np.tile(t_norm, interpolant_order))
        return y0 + timestep * np.dot(interpolant_coefficients, t_norm_powers)

class RK45:
    '''
    Explicit 5th order Runge-Kutta with 4th order error estimate. Coefficients
    and interpolant coefficients are from [1].

    [1] Ch. Tsitorasm "Runge-Kutta pairs of order 5(4) satisfying only the first
        column simplifying assumption", Computers and Mathematics with
        Applications, Vol. 62, No. 2, pp. 770-775, 2011.
    '''
    order = 5
    error_estimate_order = 4
    stages: int = 6
    # weights_a: npt.NDArray[float] = np.array([
    #     [0, 0, 0, 0, 0, 0],
    #     [0.161, 0, 0, 0, 0, 0],
    #     [-0.008480655492356989, 0.335480655492357, 0, 0, 0, 0],
    #     [2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0, 0, 0],
    #     [5.325864828439257, -11.748883564062828, 7.4955393428898365,
    #         -0.09249506636175525, 0, 0],
    #     [5.86145544294642, -12.92096931784711, 8.159367898576159,
    #         -0.071584973281401, -0.028269050394068383, 0]])
    
    # weights_b: npt.NDArray[float] = np.array([
    #     0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742,
    #     -3.290069515436081, 2.324710524099774])
    
    # weights_bhat: npt.NDArray[float] = np.array([
    #     -0.001780011052226, -0.000816434459657, 0.007880878010262,
    #     -0.144711007173263, 0.582357165452555, -0.458082105929187, 1/66])

    # weights_c: npt.NDArray[float] = np.array([0, 0.161, 0.327, 0.9,
    #     0.9800255409045097, 1])
    
    # # Order 5 - Order 4 in [1].
    # # t = np.zeros_like(weights_bhat)
    # # t[:-1] = weights_b
    # # weights_error = weights_bhat - t
    
    # Tsitsoras error estimate not working.
    # weights_error: npt.NDArray[float] = np.array([
    #     0.0946807557658392, 0.009183565540343, 0.487770528424761,
    #     1.23429756693048, 02.70771234998352, 1.86662841817058, 1/66])
    
    # # Expand interpolant in [1].
    # interpolant_order = 4
    # weights_interpolant: npt.NDArray[float] = np.array([
    #     [1.0, -2.763706197274826, 2.9132554618219126, -1.0530884977290216],
    #     [0.0, 0.1317, -0.2234, 0.1017],
    #     [0.0, 3.9302962368947516, -5.941033872131505, 2.490627285651253],
    #     [0.0, -12.411077166933676, 30.33818863028232, -16.548102889244902],
    #     [0.0, 37.50931341651104, -88.1789048947664, 47.37952196281928],
    #     [0.0, -27.896526289197286, 65.09189467479366, -34.87065786149661],
    #     [0.0, 1.5, -4.0, 2.5]
    # ])

    weights_a: npt.NDArray[float] = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]])
    
    # \hat{b}_i from [1].
    weights_b: npt.NDArray[float] = np.array(
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    
    weights_c: npt.NDArray[float] = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    
    # b_i - \hat{b}_i from [1]
    weights_error: npt.NDArray[float] = np.array(
        [-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])

    interpolant_order = 4
    weights_interpolant: npt.NDArray[float] = np.array([
        [1, -8048581381/2820520608, 8663915743/2820520608,
         -12715105075/11282082432],
        [0, 0, 0, 0],
        [0, 131558114200/32700410799, -68118460800/10900136933,
         87487479700/32700410799],
        [0, -1754552775/470086768, 14199869525/1410260304,
         -10690763975/1880347072],
        [0, 127303824393/49829197408, -318862633887/49829197408,
         701980252875 / 199316789632],
        [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
        [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])


    # Timestep control exponents.
    safety_factor = 0.9
    exponent_grow = -1 / (error_estimate_order)
    exponent_shrink = -1 / (error_estimate_order - 1)
    min_shrink = 0.2
    max_grow = 10.0

    REACHED_MAX_ITERATIONS = "Reached max iterations"

    __slots__ = ("time", "state_vector", "state_vector_derivative",
        "ode_rhs", "ode_rhs_func", "min_step", "max_step",
        "absolute_tolerance", "relative_tolerance", "max_iterations",
        "use_adaptive_timestep", "timestep", "rk_stages", "error_normalised",
        "stop_condition", "interpolant_coefficients")

    def __init__(self, time, state_vector, ode_rhs_func,
        min_step: float, max_step: float, absolute_tolerance: float,
        relative_tolerance: float, max_iterations: int,
        use_adaptive_timestep: bool):
        '''
          
        '''
        self.time = float(time)
        self.state_vector = np.array(state_vector)
        self.state_vector_derivative = np.zeros_like(self.state_vector)
        self.ode_rhs_func = ode_rhs_func
        self.min_step = float(min_step)
        self.max_step = float(max_step)
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.max_iterations = int(max_iterations)
        self.use_adaptive_timestep = use_adaptive_timestep

        self.timestep = 0.0
        self.rk_stages = np.zeros((self.stages + 1, self.state_vector.size))
        self.error_normalised = 0.0
        self.stop_condition = ""
        self.interpolant_coefficients = np.zeros((self.state_vector.size,
            self.interpolant_order))
        
        self._set_initial_timestep()

    def _norm(self, state_vector):
        '''
        Return Harrier norm of state vector.
        '''
        atol = self.absolute_tolerance
        rtol = self.relative_tolerance
        return np.linalg.norm(atol + rtol * state_vector) / state_vector.size

    def _set_initial_timestep(self):
        '''
        Estimate initial timestep for ODE solver using algorithm from [1].
        
        [1] E. Hairer, S. P. Norsett, G. Wanner, "Solving Ordinary Differential
            Equations I: Nonstiff Problems"
        '''
        t0 = self.time
        y0 = self.state_vector
        d0 = self._norm(y0)
        f0 = self.ode_rhs_func(t0, y0)
        d1 = self._norm(f0)

        if d0 < 1e-5 or d1 < 1e-5:
            dt0 = 1e-5
        else:
            dt0 = 0.01 * (d0 / d1)

        f1 = self.ode_rhs_func(t0 + dt0, y0 + dt0 * f0)
        d2 = self._norm((f1 - f0) / dt0)
        dmax = max(d1, d2)

        if dmax <= 1e-15:
            dt1 = max(1e-6, 1e-3 * dt0)
        else:
            dt1 = (0.01 / dmax)**(1 / (self.order + 1))
        
        self.timestep = min(100 * dt0, dt1, self.max_step)
        self.ode_rhs = f0

    def _step(self, timestep: float) -> bool:
        '''
        Take an explicit Runge Kutta step for given timestep.
        '''
        t, y = self.time, self.state_vector
        K = self.rk_stages

        # If first stage is same as last.
        K[0, :] = self.ode_rhs

        # Calculate timestep.
        for stage, (a, c) in enumerate(zip(self.weights_a[1:], self.weights_c[1:]), start=1):
            dy = timestep * np.dot(K[:stage].T, a[:stage])
            K[stage] = self.ode_rhs_func(t + timestep * c, y + dy)
        
        # Calculate new proposed position.
        t_new = t + timestep
        dy_dt = np.dot(K[:-1].T, self.weights_b)
        y_new = y + timestep * dy_dt
        K[-1] = self.ode_rhs_func(t_new, y_new)

        return t_new, y_new, dy_dt

    def _get_error_estimate(self, timestep, state_vector):
        '''
        Get estimate of error in step normalised to desired error.
        '''
        # Calculate error limit we will tolerate.
        atol = self.absolute_tolerance
        rtol = self.relative_tolerance
        ymax = np.maximum(self.state_vector, state_vector)
        error_limit = atol + rtol * ymax

        # Calculate the normalised error, the ratio of estimated error to
        # the limit. Use the Harier norm which also divides by the length of
        # the state vector.
        error_estimate = timestep * np.dot(self.rk_stages.T, self.weights_error)
        error_normalised = (sum((error_estimate / error_limit)**2)**0.5
            / ymax.shape[0])

        return error_normalised

    def stop_reached_min_timestep(self, timestep, timestep_min) -> str:
        msg = f"Timestep too small: {timestep:5.3e} < {timestep_min:5.3e}"
        logger.error(msg)
        return msg

    def stop_reached_max_iterations(self):
        msg = f"Reached max iterations for step: {self.max_iterations}"
        logger.error(msg)
        return msg

    def step(self) -> bool:
        '''
        Advance ODE system by a timestep chosen based on estimated error.
        '''
        dt_new = self.timestep

        for step_attempt in range(1, self.max_iterations + 1):
            if (dt_new < self.min_step):
                # Timestep proposed is too small.
                self.stop_condition = self.stop_reached_min_timestep(dt_new,
                    self.min_step)
                return

            # Attempt step with given timestep.
            t_new, y_new, dy_dt_new = self._step(dt_new)
            
            if self.use_adaptive_timestep:
                # Check if step was ok and make timestep adjustments using a
                # proportional error controller.
                error_normalised = self._get_error_estimate(dt_new, y_new)

                if error_normalised > 1:
                    # Error too large, reduce timestep.
                    accept_step = False
                    factor = self.safety_factor * error_normalised**self.exponent_shrink
                else:
                    # Step ok, can increase timestep.
                    accept_step = True

                    if error_normalised == 0:
                        factor = self.max_grow
                    else:
                        factor = self.safety_factor * error_normalised**self.exponent_grow

                # Adjust proposed timestep, clipping factor between min and max.
                factor = min(max(factor, self.min_shrink), self.max_grow)
                dt_new = min(dt_new * factor, self.max_step)
            else:
                accept_step = True

            # If step is ok break out of loop.
            if accept_step:
                self.time = t_new
                self.state_vector[...] = y_new
                self.state_vector_derivative[...] = dy_dt_new
                self.ode_rhs = self.rk_stages[-1]
                self.timestep = dt_new
                self.interpolant_coefficients[...] = \
                    self._calculate_interpolant_coefficients()
    
                break

        if step_attempt == self.max_iterations:
            # Reached maximum number of iterations without finding an
            # acceptable step.
            self.stop_condition = self.stop_reached_max_iterations()
            return
    
    def _calculate_interpolant_coefficients(self):
        '''
        Calculate coefficients for polynomial interpolation of the solution
        value in the step and set in the cache.
        '''
        interpolant_coefficients = np.dot(self.rk_stages.T,
            self.weights_interpolant)
        return interpolant_coefficients
    