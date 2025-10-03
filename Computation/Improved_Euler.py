import numpy as np
from HH_ODE import hh_ode
import matplotlib.pyplot as plt

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

    t_span = (0, 50)  # ms
    y0 = np.array([-65, 0.05, 0.6, 0.32])  # [V0, m0, h0, n0]
    n_steps = 1000

    def I_ext(t):
        return 10.0 if 10 <= t <= 40 else 0.0

    # Solve the Hodgkin-Huxley equations
    t, solution = improved_euler_method(hh_ode, t_span, y0, n_steps, I_ext)

    # Extract variables from solution
    V = solution[:, 0]  # Membrane potential
    m = solution[:, 1]  # Sodium activation
    h = solution[:, 2]  # Sodium inactivation  
    n = solution[:, 3]  # Potassium activation

    # Create applied current for plotting
    I_applied = np.array([I_ext(time) for time in t])

    # Plot results
    plt.figure(figsize=(12, 10))

    # Plot 1: Membrane Potential
    plt.subplot(3, 1, 1)
    plt.plot(t, V, 'b-', linewidth=2)
    plt.title('Hodgkin-Huxley Model - Improved Euler Method', fontsize=14)
    plt.ylabel('Membrane Potential (mV)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(['V(t)'], loc='upper right')

    # Plot 2: Gating Variables
    plt.subplot(3, 1, 2)
    plt.plot(t, m, 'r-', label='m (Na activation)')
    plt.plot(t, h, 'g-', label='h (Na inactivation)')
    plt.plot(t, n, 'b-', label='n (K activation)')
    plt.ylabel('Gating Variables', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # Plot 3: Applied Current
    plt.subplot(3, 1, 3)
    plt.plot(t, I_applied, 'k-', linewidth=2)
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Applied Current (μA/cm²)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(['I_ext(t)'], loc='upper right')

    plt.tight_layout()
    plt.show()