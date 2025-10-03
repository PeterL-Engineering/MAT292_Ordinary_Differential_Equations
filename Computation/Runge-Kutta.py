import numpy as np
from HH_ODE import hh_ode

def runge_kutta(f, t_span, y0, n_steps):
    """
    Function Description:
        - Solves a system of ordinary differential equations using the 4th-order Runge-Kutta method
        - Provides higher accuracy than Euler methods by using weighted averages of multiple slope estimates
        - Specifically designed for systems with 4 state variables (e.g., Hodgkin-Huxley model)
    
    Parameters:
        - f (callable): Function that computes the derivatives of the system
            - Signature: f(t, y) -> array_like
            - t: Current time
            - y: Current state vector [V, m, h, n] for Hodgkin-Huxley
            - Returns: Array of derivatives [dV/dt, dm/dt, dh/dt, dn/dt]
        - t_span (tuple): Time range for integration as (start_time, end_time) in milliseconds
        - y0 (array_like): Initial conditions vector [V0, m0, h0, n0] where:
            - V0: Initial membrane potential (mV)
            - m0: Initial sodium activation gating variable
            - h0: Initial sodium inactivation gating variable
            - n0: Initial potassium activation gating variable
        - n_steps (int): Number of time steps to divide the integration interval
    
    Returns:
        - t (numpy.ndarray): Array of time points from t_span[0] to t_span[1]
            - Shape: (n_steps + 1,)
        - y (numpy.ndarray): Solution matrix containing the approximated state variables at each time point
            - Shape: (n_steps + 1, 4)
            - y[:, 0]: Membrane potential V over time
            - y[:, 1]: Sodium activation m over time
            - y[:, 2]: Sodium inactivation h over time
            - y[:, 3]: Potassium activation n over time
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    y = np.full((len(t), 4), np.nan)  # N×4 solution matrix
    y[0, :] = y0  # Initial conditions [V0, m0, h0, n0]
    
    for i in range(len(t) - 1):
        h = t[i+1] - t[i]
        
        k1 = f(t[i], y[i, :])
        k2 = f(t[i] + h/2, y[i, :] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i, :] + h/2 * k2)
        k4 = f(t[i] + h, y[i, :] + h * k3)
        
        y[i+1, :] = y[i, :] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

# if __name__ == "__main__":
    