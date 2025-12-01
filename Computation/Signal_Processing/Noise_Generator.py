import sys
import os

# Add all necessary paths
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
testing_dir = os.path.join(current_dir, '../Testing')
numerical_methods_dir = os.path.join(current_dir, '../Numerical_Methods')

sys.path.extend([current_dir, parent_dir, testing_dir, numerical_methods_dir])

import numpy as np
import matplotlib.pyplot as plt

from Numerical_Methods.HH_ODE import hh_ode
from Numerical_Methods.Applied_Current import I_ext_burst, I_ext_double_pulse, I_ext_fm, I_ext_noisy, I_ext_ramp, I_ext_sinusoidal
from Numerical_Methods.Improved_Euler import improved_euler_method
from Testing import plot_hodgkin_huxley_results

def gaussian_noise(input_signal, mean=0.0, std_dev=1.0, seed=None):
    """
    Function Description:
        - Adds Gaussian noise to a solution array of the HH system
        - Generates random noise from a normal distribution and adds it to the input signal
    
    Parameters:
        - input_signal (array_like): Input signal or array to which noise will be added
        - mean (float): Mean of the Gaussian noise distribution (default: 0.0)
        - std_dev (float): Standard deviation of the Gaussian noise distribution (default: 1.0)
        - seed (int, optional): Random seed for reproducible noise generation
    
    Returns:
        - noisy_signal (numpy.ndarray): Input signal with added Gaussian noise
    """

    if std_dev < 0:
        raise ValueError("Standard deviation must be non-negative")
    
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.normal(mean, std_dev, input_signal.shape)
    else:
        noise = np.random.normal(mean, std_dev, input_signal.shape)
    
    noisy_signal = input_signal + noise

    return noisy_signal

def amplitude_modulation(input_signal, modTaper, modWidth, modMinRelAmplitude, dt=0.1):
    """
    Function Description:
        - Adds amplitude modulation to input signal
        - Creates a modulation window with quadratic taper profiles and applies it randomly within the signal
    
    Parameters:
        - input_signal (array_like): Input signal to be modulated
        - modTaper (float): Duration of the taper regions in seconds
        - modWidth (float): Total duration of the modulation window in seconds
        - modMinRelAmplitude (float): Minimum relative amplitude during the flat modulation period (0.0 to 1.0)
        - dt (float): Time step between samples in seconds (default: 0.1)
    
    Returns:
        - modulated_signal (numpy.ndarray): Amplitude-modulated copy of the input signal
    """
    modulated_signal = input_signal.copy()
    n_samples = len(input_signal)
    
    flat_duration = modWidth - 2 * modTaper
    if flat_duration < 0:
        raise ValueError("modWidth must be at least 2 * modTaper")
    
    max_start = n_samples - modWidth
    if max_start <= 0:
        raise ValueError("Signal too short for specified modulation width")
    
    start_idx = np.random.randint(0, max_start)
    
    # Create taper profiles (quadratic)
    up_taper_indices = np.arange(modTaper)
    up_taper = modMinRelAmplitude + (1 - modMinRelAmplitude) * (up_taper_indices / modTaper) ** 2
    
    down_taper_indices = np.arange(modTaper)
    down_taper = 1.0 - (1 - modMinRelAmplitude) * (down_taper_indices / modTaper) ** 2
    
    # Apply modulation
    start_taper_start = start_idx
    start_taper_end = start_idx + modTaper
    modulated_signal[start_taper_start:start_taper_end] *= up_taper
    
    flat_start = start_idx + modTaper
    flat_end = start_idx + modTaper + flat_duration
    modulated_signal[flat_start:flat_end] *= modMinRelAmplitude
    
    end_taper_start = start_idx + modTaper + flat_duration
    end_taper_end = start_idx + modWidth
    modulated_signal[end_taper_start:end_taper_end] *= down_taper
    
    return modulated_signal

def ocular_artifact(input_signal, dt=0.1):
    """
    Function Description:
        - Adds large amplitude spikes to input signal mimicking ocular artifacts
        - Generates triangular-shaped artifacts with random placement in the signal
    
    Parameters:
        - input_signal (array_like): Input signal to which ocular artifacts will be added
        - dt (float): Time step between samples in seconds (default: 0.1)
    
    Returns:
        - output_signal (numpy.ndarray): Copy of input signal with added ocular artifacts
    """
    artifact_duration = 1.0  # seconds of artifact duration
    artifact_length = int(artifact_duration / dt)  # convert to samples
    n_samples = len(input_signal)
    max_start = n_samples - artifact_length
    
    if max_start <= 0:
        raise ValueError("Signal too short to add ocular artifact")
    
    start_idx = np.random.randint(0, max_start)
    artifact_amplitude = 1.5 * np.max(np.abs(input_signal))
    
    # Create triangular artifact shape
    artifact = np.zeros(artifact_length)
    half_length = artifact_length // 2
    
    # Build rising edge
    for i in range(half_length):
        artifact[i] = artifact_amplitude * (i / half_length)
    
    # Build falling edge  
    for i in range(half_length, artifact_length):
        artifact[i] = artifact_amplitude * ((half_length - i - 1) / half_length)
    
    # Add artifact to signal
    output_signal = input_signal.copy()
    output_signal[start_idx:start_idx + artifact_length] += artifact
    
    return output_signal

def plot_noise_comparison(t, V_original, V_gaussian, V_amplitude_mod, V_ocular, V_combined, 
                         figsize=(15, 12), show_stats=True):
    """
    Function Description:
        - Creates a comprehensive comparison plot of original signal vs various noise types
        - Displays all noise effects in a multi-panel layout with statistics
    
    Parameters:
        - t (array_like): Time array for x-axis
        - V_original (array_like): Original membrane potential signal
        - V_gaussian (array_like): Signal with Gaussian noise
        - V_amplitude_mod (array_like): Signal with amplitude modulation
        - V_ocular (array_like): Signal with ocular artifact
        - V_combined (array_like): Signal with all noise effects combined
        - figsize (tuple): Figure size (width, height) in inches
        - show_stats (bool): Whether to print noise statistics to console
    
    Returns:
        - fig (matplotlib.figure.Figure): The created figure object
        - axs (numpy.ndarray): Array of axes objects
    """
    
    # Create figure and subplots
    fig, axs = plt.subplots(5, 1, figsize=figsize)
    
    # Plot configurations
    plot_configs = [
        {'data': V_original, 'color': 'b-', 'title': 'Original Hodgkin-Huxley Membrane Potential', 'legend': ['Original']},
        {'data': V_gaussian, 'color': 'r-', 'title': 'With Gaussian Noise (std=10.0)', 'legend': ['Gaussian Noise']},
        {'data': V_amplitude_mod, 'color': 'g-', 'title': 'With Amplitude Modulation', 'legend': ['Amplitude Modulated']},
        {'data': V_ocular, 'color': 'purple', 'title': 'With Ocular Artifact', 'legend': ['Ocular Artifact']},
        {'data': V_combined, 'color': 'orange', 'title': 'Combined Noise Effects', 'legend': ['Combined Noise']}
    ]
    
    # Create each subplot
    for i, config in enumerate(plot_configs):
        axs[i].plot(t, config['data'], config['color'], linewidth=1.5)
        axs[i].set_title(config['title'], fontsize=12)
        axs[i].set_ylabel('V (mV)')
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(config['legend'], loc='upper right')
        
        # Add x-label only to bottom plot
        if i == len(plot_configs) - 1:
            axs[i].set_xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics if requested
    if show_stats:
        print_noise_statistics(V_original, V_gaussian, V_amplitude_mod, V_ocular, V_combined)
    
    return fig, axs

def print_noise_statistics(V_original, V_gaussian, V_amplitude_mod, V_ocular, V_combined):
    """
    Function Description:
        - Prints statistical comparison of different noise effects
        - Shows range and basic statistics for each signal type
    
    Parameters:
        - V_original (array_like): Original membrane potential signal
        - V_gaussian (array_like): Signal with Gaussian noise
        - V_amplitude_mod (array_like): Signal with amplitude modulation
        - V_ocular (array_like): Signal with ocular artifact
        - V_combined (array_like): Signal with all noise effects combined
    """
    print("\nNoise Statistics:")
    print(f"Original signal range: {np.min(V_original):.2f} to {np.max(V_original):.2f} mV")
    print(f"Gaussian noise - Range: {np.min(V_gaussian):.2f} to {np.max(V_gaussian):.2f} mV")
    print(f"Amplitude mod - Range: {np.min(V_amplitude_mod):.2f} to {np.max(V_amplitude_mod):.2f} mV")
    print(f"Ocular artifact - Range: {np.min(V_ocular):.2f} to {np.max(V_ocular):.2f} mV")
    print(f"Combined noise - Range: {np.min(V_combined):.2f} to {np.max(V_combined):.2f} mV")

def apply_all_noise_types(V, dt, gaussian_std=10.0, modTaper=100, modWidth=500, 
                         modMinRelAmplitude=0.3, seed=42):
    """
    Function Description:
        - Applies all noise types to a signal and returns individual and combined results
        - Convenience function for consistent noise application
    
    Parameters:
        - V (array_like): Original signal
        - dt (float): Time step
        - gaussian_std (float): Standard deviation for Gaussian noise
        - modTaper (int): Taper duration for amplitude modulation
        - modWidth (int): Total modulation width
        - modMinRelAmplitude (float): Minimum relative amplitude
        - seed (int): Random seed for reproducibility
    
    Returns:
        - tuple: (V_gaussian, V_amplitude_mod, V_ocular, V_combined)
    """
    # Apply individual noise types
    V_gaussian = gaussian_noise(V, mean=0.0, std_dev=gaussian_std, seed=seed)
    V_amplitude_mod = amplitude_modulation(V, modTaper, modWidth, modMinRelAmplitude, dt)
    V_ocular = ocular_artifact(V, dt)
    
    # Apply combined noise in optimal order
    V_combined = V.copy()
    V_combined = amplitude_modulation(V_combined, modTaper, modWidth, modMinRelAmplitude, dt)
    V_combined = gaussian_noise(V_combined, mean=0.0, std_dev=gaussian_std, seed=seed)
    V_combined = ocular_artifact(V_combined, dt)
    
    return V_gaussian, V_amplitude_mod, V_ocular, V_combined

if __name__ == "__main__":
    # Simulation parameters
    t_span = (0, 100)  # ms
    n_steps = 2000
    
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

    I_ext = I_ext_burst
    np.random.seed(42)
    
    # Solve the Hodgkin-Huxley equations
    t, solution = improved_euler_method(hh_ode, t_span, y0_k_activated, n_steps, I_ext)

    # Extract variables from solution
    V = solution[:, 0]
    m = solution[:, 1]
    h = solution[:, 2]
    n = solution[:, 3]

    # Create applied current for plotting
    I_applied = np.array([I_ext(time) for time in t])
    
    # Calculate time step for noise functions
    dt = (t_span[1] - t_span[0]) / n_steps
    
    # Apply all noise types using the convenience function
    V_gaussian, V_amplitude_mod, V_ocular, V_combined = apply_all_noise_types(
        V, dt, gaussian_std=10.0, modTaper=100, modWidth=500, 
        modMinRelAmplitude=0.3, seed=42
    )
    
    # Plot the comparison using the modular plotting function
    plot_noise_comparison(t, V, V_gaussian, V_amplitude_mod, V_ocular, V_combined)