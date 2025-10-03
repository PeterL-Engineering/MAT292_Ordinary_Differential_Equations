import numpy as np
from HH_ODE import hh_ode
import matplotlib.pyplot as plt

def euler_method(f, t_span, y0, n_steps, *args):
    """
    Function Description:
        - Solves a system of ordinary differential equations using Euler's method
        - Calculates the tangent slope at each time step to approximate the solution
    
    Parameters:
        - f (callable): Function that computes the derivatives of the system
            - Signature: f(t, y, *args) -> array_like
        - t_span (tuple): Time range for integration as (start_time, end_time)
        - y0 (array_like): Initial conditions vector
        - n_steps (int): Number of time steps to divide the integration interval
        - *args: Additional arguments to pass to the ODE function f
    
    Returns:
        - t (numpy.ndarray): Array of time points from t_span[0] to t_span[1]
        - y (numpy.ndarray): Solution matrix containing approximated state variables
    """
    
    # Create time array
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    dt = t[1] - t[0]
    
    # Initialize solution array
    y = np.full((len(t), len(y0)), np.nan)
    y[0, :] = y0

    # Iteratively approximate the solution using Euler's method
    for i in range(len(t) - 1):
        # Calculate derivative at current point and update
        derivative = f(t[i], y[i, :], *args)
        y[i + 1, :] = y[i, :] + dt * derivative

    return t, y


if __name__ == "__main__":

    t_span = (0, 50)  # ms
    y0 = np.array([-65, 0.05, 0.6, 0.32])  # [V0, m0, h0, n0]
    n_steps = 1000
    
    def I_ext(t):
        return 10.0 if 10 <= t <= 40 else 0.0
    
    # Solve using Euler method
    t, solution = euler_method(hh_ode, t_span, y0, n_steps, I_ext)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t, solution[:, 0])
    plt.title('Membrane Potential V(t)')
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    
    plt.subplot(2, 2, 2)
    plt.plot(t, solution[:, 1], label='m')
    plt.plot(t, solution[:, 2], label='h')
    plt.plot(t, solution[:, 3], label='n')
    plt.title('Gating Variables')
    plt.xlabel('Time (ms)')
    plt.ylabel('Gating variable')
    plt.legend()
    
    plt.tight_layout()
    plt.show()