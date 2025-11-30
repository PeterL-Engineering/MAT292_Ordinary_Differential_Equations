import numpy as np

"""
The following functions are a set of functions that represent
activation current when computing HH equations
"""

# 1. Ramp current (simulates gradually increasing stimulus)
def I_ext_ramp(t):
    """
    Function Description:
        - Generates a ramp current stimulus for Hodgkin-Huxley simulations
        - Provides linearly increasing current followed by return to baseline
    
    Parameters:
        - t (float): Time point at which to evaluate the current
    
    Returns:
        - current (float): External current value at time t
    """
    if t < 10:
        return 0.0
    elif t < 60:
        return 0.5 * (t - 10)  # Linear ramp
    else:
        return 0.0

# 2. Sinusoidal current (simulates rhythmic input)
def I_ext_sinusoidal(t):
    """
    Function Description:
        - Generates a sinusoidal current stimulus for Hodgkin-Huxley simulations
        - Provides rhythmic oscillatory input with constant frequency and amplitude
    
    Parameters:
        - t (float): Time point at which to evaluate the current
    
    Returns:
        - current (float): External current value at time t
    """
    if t < 10:
        return 0.0
    elif t <= 80:
        return 8.0 + 4.0 * np.sin(2 * np.pi * 0.05 * (t - 10))  # 0.05 Hz oscillation
    else:
        return 0.0

# 3. Burst pattern (simulates natural bursting behavior)
def I_ext_burst(t):
    """
    Function Description:
        - Generates burst-pattern current stimulus for Hodgkin-Huxley simulations
        - Creates two distinct burst patterns with different timing characteristics
    
    Parameters:
        - t (float): Time point at which to evaluate the current
    
    Returns:
        - current (float): External current value at time t
    """
    if t < 10:
        return 0.0
    elif 10 <= t < 40:
        # Burst pattern: high frequency pulses
        pulse_phase = (t - 10) % 8  # 8ms period
        return 12.0 if pulse_phase < 4 else 2.0  # 4ms on, 4ms off
    elif 40 <= t < 70:
        # Different burst pattern
        pulse_phase = (t - 40) % 12  # 12ms period
        return 15.0 if pulse_phase < 3 else 1.0  # 3ms on, 9ms off
    else:
        return 0.0

# 4. Noisy current (more biologically realistic)
def I_ext_noisy(t):
    """
    Function Description:
        - Generates noisy current stimulus for Hodgkin-Huxley simulations
        - Provides base current with Gaussian noise for biological realism
    
    Parameters:
        - t (float): Time point at which to evaluate the current
    
    Returns:
        - current (float): External current value at time t (non-negative)
    """
    if t < 10:
        return 0.0
    elif t <= 60:
        # Base current with noise
        base_current = 8.0
        noise = np.random.normal(0, 2.0)  # Gaussian noise
        return max(0, base_current + noise)  # Ensure non-negative
    else:
        return 0.0

# 5. Double pulse with varying intervals
def I_ext_double_pulse(t):
    """
    Function Description:
        - Generates double-pulse current stimulus for Hodgkin-Huxley simulations
        - Provides three distinct pulses with varying amplitudes and durations
    
    Parameters:
        - t (float): Time point at which to evaluate the current
    
    Returns:
        - current (float): External current value at time t
    """
    if 10 <= t < 20:
        return 12.0
    elif 35 <= t < 45:
        return 15.0
    elif 65 <= t < 70:
        return 8.0
    else:
        return 0.0

# 6. Frequency modulation (changing frequency over time)
def I_ext_fm(t):
    """
    Function Description:
        - Generates frequency-modulated current stimulus for Hodgkin-Huxley simulations
        - Provides sinusoidal current with linearly increasing frequency over time
    
    Parameters:
        - t (float): Time point at which to evaluate the current
    
    Returns:
        - current (float): External current value at time t
    """
    if t < 10:
        return 0.0
    elif t <= 80:
        # Frequency increases over time
        freq = 0.02 + 0.001 * (t - 10)  # Increasing frequency
        return 10.0 + 5.0 * np.sin(2 * np.pi * freq * (t - 10))
    else:
        return 0.0