import numpy as np

def hh_ode(t, y, I_ext):
    """
    Function Description:
        - Computes the derivatives of the Hodgkin-Huxley neuron model
        - Models the dynamics of membrane potential and ion channel gating variables
        - Represents a system of four coupled non-linear ordinary differential equations
    
    Parameters:
        - t (float): Current time in milliseconds
        - y (array_like): Array of current state variables [V, m, h, n] where:
            - V: Membrane potential in mV
            - m: Sodium activation gating variable (dimensionless)
            - h: Sodium inactivation gating variable (dimensionless)  
            - n: Potassium activation gating variable (dimensionless)
        - I_ext (callable): External current input function I_ext(t) that returns current in μA/cm^2 at time t
    
    Returns:
        - dydt: Array of derivatives [dV/dt, dm/dt, dh/dt, dn/dt] representing:
            - dV/dt: Rate of change of membrane potential in mV/ms
            - dm/dt: Rate of change of sodium activation in ms^-1
            - dh/dt: Rate of change of sodium inactivation in ms^-1
            - dn/dt: Rate of change of potassium activation in ms^-1
    """
    # Hodgkin-Huxley parameters (standard values)
    g_Na = 120.0  # Sodium conductance (mS/cm^2)
    g_K = 36.0    # Potassium conductance (mS/cm^2)
    g_L = 0.3     # Leak conductance (mS/cm^2)
    E_Na = 50.0   # Sodium reversal potential (mV)
    E_K = -77.0   # Potassium reversal potential (mV)
    E_L = -54.387 # Leak reversal potential (mV)
    C_m = 1.0     # Membrane capacitance (μF/cm^2)
    
    V, m, h, n = y
    
    # Hodgkin-Huxley currents
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K) 
    I_L = g_L * (V - E_L)
    
    # Membrane potential derivative
    dVdt = (I_ext(t) - I_Na - I_K - I_L) / C_m
    
    # Gate dynamics (non-linear functions of V)
    alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta_m = 4.0 * np.exp(-(V + 65) / 18)
    dmdt = alpha_m * (1 - m) - beta_m * m
    
    alpha_h = 0.07 * np.exp(-(V + 65) / 20)
    beta_h = 1.0 / (1 + np.exp(-(V + 35) / 10))
    dhdt = alpha_h * (1 - h) - beta_h * h
    
    alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    beta_n = 0.125 * np.exp(-(V + 65) / 80)
    dndt = alpha_n * (1 - n) - beta_n * n
    
    return np.array([dVdt, dmdt, dhdt, dndt])