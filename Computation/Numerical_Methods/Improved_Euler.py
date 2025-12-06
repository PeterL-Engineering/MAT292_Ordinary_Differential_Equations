import sys
import os

sys.path.append(os.path.dirname(__file__))  # Current directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '../Testing'))  # Testing directory

import numpy as np
from HH_ODE import hh_ode
from Applied_Current import I_ext_burst, I_ext_double_pulse, I_ext_fm, I_ext_noisy, I_ext_ramp, I_ext_sinusoidal
from Testing import plot_hodgkin_huxley_results

def improved_euler_method(f, t_span, y0, n_steps, *args):
    """
    Function Description:
        - Solves a system of ordinary differential equations using Improved Euler's method (Heun's method)
        - This method uses an average of the slope at the beginning and estimated end of the interval
        - Works for both linear and nonlinear systems
    
    Parameters:
        - f (callable): Function that computes the derivatives of the system
            - Signature: f(t, y, *args) -> array_like
            - t: Current time
            - y: Current state vector
            - *args: Additional arguments for the ODE function (e.g., I_ext for Hodgkin-Huxley)
        - t_span (tuple): Time range for integration as (start_time, end_time)
        - y0 (array_like): Initial conditions vector
        - n_steps (int): Number of time steps to divide the integration interval
        - *args: Additional arguments to pass to the ODE function f
    
    Returns:
        - t (numpy.ndarray): Array of time points from t_span[0] to t_span[1]
            - Shape: (n_steps + 1,)
        - y (numpy.ndarray): Solution matrix containing the approximated state variables at each time point
            - Shape: (n_steps + 1, len(y0))
    """
    
    # Create time array
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    dt = t[1] - t[0]
    
    # Initialize solution array
    y = np.full((len(t), len(y0)), np.nan)
    y[0, :] = y0

    # Iteratively approximate the solution using Improved Euler
    for i in range(len(t) - 1):
        # Step 1: Predictor (Euler's method)
        k1 = f(t[i], y[i, :], *args)
        predictor = y[i, :] + dt * k1
        
        # Step 2: Corrector (average of slopes)
        k2 = f(t[i] + dt, predictor, *args)  # Slope at predicted point
        avg_slope = (k1 + k2) / 2
        
        # Update solution with corrected value
        y[i + 1, :] = y[i, :] + dt * avg_slope

    return t, y

if __name__ == "__main__":
    t_span = (0, 100)  # ms
    n_steps = 1900 # Do not go below 19x t_span
    
    # 1. Depolarized start (simulating recent synaptic input)
    y0_depolarized = np.array([-45, 0.3, 0.4, 0.5])  # [V0, m0, h0, n0]
    
    # 2. Hyperpolarized start (recent inhibition)
    y0_hyperpolarized = np.array([-80, 0.01, 0.8, 0.2])  # [V0, m0, h0, n0]
    
    # 3. Post-spike state (refractory period)
    y0_post_spike = np.array([20, 0.9, 0.1, 0.8])  # Just after an action potential
    
    # 4. Sodium channel inactivated (simulating some drugs or pathology)
    y0_na_inactivated = np.array([-65, 0.05, 0.1, 0.32])  # Low h value
    
    # 5. Potassium channel activated (increased K+ conductance)
    y0_k_activated = np.array([-65, 0.05, 0.6, 0.8])  # High n value
    
    # 6. Mixed state - partially activated
    y0_mixed = np.array([-55, 0.2, 0.3, 0.4])
    
    # 7. Resting but with different gating variable combinations
    y0_alternative_rest = np.array([-65, 0.05, 0.5, 0.3])
    
    # 8. Near threshold state
    y0_near_threshold = np.array([-55, 0.1, 0.5, 0.35])

    I_ext = I_ext_burst  # Select current pattern
    initial_condition = y0_k_activated
    np.random.seed(42)
    
    # Solve the Hodgkin-Huxley equations using Runge-Kutta method
    t, solution = improved_euler_method(hh_ode, t_span, initial_condition, n_steps, I_ext)

    # Extract variables from solution
    V = solution[:, 0]
    m = solution[:, 1]
    h = solution[:, 2]
    n = solution[:, 3]

    # Create applied current for plotting
    I_applied = np.array([I_ext(time) for time in t])

    # Plot solution curves
    plot_hodgkin_huxley_results(t, V, m, h, n, I_applied, I_ext)