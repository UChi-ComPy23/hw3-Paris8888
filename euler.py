"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn

## Problem 0 Part A (20 points)

"""
Defintions for problem 0
"""

class ForwardEulerOutput(DenseOutput): 
    """
    Dense output for the Forward Euler method (linear interpolation).
    """
    # The constructor method, called when a new instance is created
    def __init__(self, t_old, t, y_old, y):
        # Call the parent class's constructor to set t_old and t_new
        super(ForwardEulerOutput, self).__init__(t_old, t)
        # Store the solution vector 'y' at the beginning of the step
        self.y_old = y_old
        # Store the solution vector 'y' at the end of the step
        self.y = y
        # Calculate and store the time step size 'h'
        self.h = t - t_old
        
        # Calculate the difference for interpolation
        # Check if the time step size is zero
        if self.h == 0:
            # If so, the difference (slope) is a zero vector
            self.y_diff = np.zeros_like(y_old)
        else:
            # This was y_diff = (y - y_old) / h,
            # which is y_diff * (t - t_old) in _call_impl.
            # Calculate the slope of the line connecting y_old and y
            self.y_diff = (self.y - self.y_old) / self.h

    # This method is called when we want to get the solution at a specific time 't'
    def _call_impl(self, t):
        """
        Evaluate the dense output at time t (scalar or array).
        """
        # Ensure the input time 't' is a NumPy array for consistent calculations
        t = np.asarray(t)
        
        # Calculate interpolation factor (time elapsed since t_old)
        # This 'theta' is the (t - t_old) part of the linear interpolation formula
        theta = t - self.t_old
        
        # Perform linear interpolation: y(t) = y_old + (t - t_old) * slope
        # Check if the requested time 't' is a single number (a scalar)
        if t.ndim == 0:
            # Perform standard linear interpolation
            y_interp = self.y_old + self.y_diff * theta
        # If 't' is an array of multiple time points
        else:
            # Use NumPy broadcasting ([:, None]) to perform the interpolation for all times 't' at once
            y_interp = self.y_old[:, None] + self.y_diff[:, None] * theta
        
        return y_interp

# Define the main solver class, inheriting from SciPy's OdeSolver
class ForwardEuler(scipy.integrate.OdeSolver):
    """
    Custom ODE solver for solve_ivp based on the Forward Euler method.
    """
    
    def __init__(self, fun, t0, y0, t_bound, vectorized, h=None, **extraneous):
        
        for key in extraneous:
            warn(f"Keyword argument '{key}' is not used by ForwardEuler solver.")
        
        super(ForwardEuler, self).__init__(fun, t0, y0, t_bound, vectorized)
        
        self.y_old = None
        
        if h is None:
            default_h = (self.t_bound - self.t0) / 100.0
            self.h_abs = np.abs(default_h) if default_h != 0 else 1e-6
        else:
            self.h_abs = np.abs(h)
        
        # Set the actual step size 'h' to use, including the correct sign (direction)
        self.h = self.direction * self.h_abs

        # Initialize the number of Jacobian evaluations (not used by Euler)
        self.njev = 0
        # Initialize the number of LU decompositions (not used by Euler)
        self.nlu = 0

    # This method performs a single integration step
    def _step_impl(self):
        """
        Performs one step of the Forward Euler method.
        """
        # Get the current time 't' from the solver's state
        t = self.t
        # Get the current solution 'y' from the solver's state
        y = self.y
        
        # Store the current 'y' as 'y_old' (needed for dense output)
        self.y_old = y
        
        h = self.h
        
        if self.direction * (t + h - self.t_bound) > 0:
            h = self.t_bound - t
        
        # We use self.y.dtype (from the numpy state array) 
        if np.abs(h) < 10 * np.finfo(self.y.dtype).eps:
            self.t = self.t_bound
            return True, None 

        # This is the core Forward Euler step:
        f_eval = self.fun(t, y) # 1. Evaluate the derivative function f(t, y)
        y_new = y + h * f_eval  # 2. Calculate new y: y(t+h) = y(t) + h * f(t, y)
        t_new = t + h           # 3. Calculate the new time t(t+h)

        # Update the solver's state with the new time
        self.t = t_new
        # Update the solver's state with the new solution
        self.y = y_new
        
        return True, None

    def _dense_output_impl(self):
        """
        Returns a dense output object covering the last step.
        """
        # Check if one step is even taken yet
        if self.y_old is None:
            raise RuntimeError("Dense output is not available before the first step.")
        # Create and return a new instance of ForwardEulerOutput class
        # This instance holds all the info needed to interpolate between t_old and t
        return ForwardEulerOutput(self.t_old, self.t, self.y_old, self.y)


