import numpy as np
import matplotlib.pyplot as plt
from HH_ODE import hh_ode
from Applied_Current import I_ext_burst, I_ext_double_pulse, I_ext_fm, I_ext_noisy, I_ext_ramp, I_ext_sinusoidal
from Graph_Solution import plot_hodgkin_huxley_results
from Improved_Euler import improved_euler_method

def gaussian_noise(input_signal, mean=0.0, std_dev=1.0, seed=None):
    """
    Adds Gaussian noise to input signal.
    """
    if std_dev < 0:
        raise ValueError("Standard deviation must be non-negative")
    
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.normal(mean, std_dev, input_signal.shape)
    else:
        noise = np.random.normal(mean, std_dev, input_signal.shape)
    
    return input_signal + noise

def amplitude_modulation(input_signal, modTaper, modWidth, modMinRelAmplitude, dt=0.1):
    """
    Adds amplitude modulation to input signal
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
    Adds large amplitude spikes to input signal mimicking ocular artifacts
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

if __name__ == "__main__":
    t_span = (0, 100)  # ms
    n_steps = 2000
    
    # Initial conditions
    y0_k_activated = np.array([-65, 0.05, 0.6, 0.8])  # High n value
    
    I_ext = I_ext_burst  # Select current pattern
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
    
    # Apply different types of noise to the membrane potential
    print("Applying noise to membrane potential...")
    
    # 1. Gaussian noise
    V_gaussian = gaussian_noise(V, mean=0.0, std_dev=10.0, seed=42)
    
    # 2. Amplitude modulation
    modTaper = 100  # samples
    modWidth = 500  # samples  
    modMinRelAmplitude = 0.3  # 30% of original amplitude
    V_amplitude_mod = amplitude_modulation(V, modTaper, modWidth, modMinRelAmplitude, dt)
    
    # 3. Ocular artifact
    V_ocular = ocular_artifact(V, dt)
    
    # 4. Combined noise (all three effects) - FIXED VERSION
    # Use the SAME parameters as individual effects and apply in optimal order
    V_combined = V.copy()
    
    # Apply amplitude modulation first (affects signal strength)
    V_combined = amplitude_modulation(V_combined, modTaper, modWidth, modMinRelAmplitude, dt)
    
    # Then add Gaussian noise (same std_dev as individual case)
    V_combined = gaussian_noise(V_combined, mean=0.0, std_dev=10.0, seed=42)
    
    # Finally add ocular artifact (large spikes that should be visible)
    V_combined = ocular_artifact(V_combined, dt)
    
    # Plot original and noisy signals
    
    # Create a comprehensive comparison plot
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Original signal
    plt.subplot(5, 1, 1)
    plt.plot(t, V, 'b-', linewidth=1.5)
    plt.title('Original Hodgkin-Huxley Membrane Potential', fontsize=12)
    plt.ylabel('V (mV)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Original'], loc='upper right')
    
    # Plot 2: Gaussian noise
    plt.subplot(5, 1, 2)
    plt.plot(t, V_gaussian, 'r-', linewidth=1.5)
    plt.title('With Gaussian Noise (std=2.0)', fontsize=12)
    plt.ylabel('V (mV)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Gaussian Noise'], loc='upper right')
    
    # Plot 3: Amplitude modulation
    plt.subplot(5, 1, 3)
    plt.plot(t, V_amplitude_mod, 'g-', linewidth=1.5)
    plt.title('With Amplitude Modulation', fontsize=12)
    plt.ylabel('V (mV)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Amplitude Modulated'], loc='upper right')
    
    # Plot 4: Ocular artifact
    plt.subplot(5, 1, 4)
    plt.plot(t, V_ocular, 'purple', linewidth=1.5)
    plt.title('With Ocular Artifact', fontsize=12)
    plt.ylabel('V (mV)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Ocular Artifact'], loc='upper right')
    
    # Plot 5: Combined noise
    plt.subplot(5, 1, 5)
    plt.plot(t, V_combined, 'orange', linewidth=1.5)
    plt.title('Combined Noise Effects', fontsize=12)
    plt.ylabel('V (mV)')
    plt.xlabel('Time (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Combined Noise'], loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Also plot individual noisy versions with full Hodgkin-Huxley plots
    print("\nGenerating detailed plots for each noise type...")
    
    # Gaussian noise version
    solution_gaussian = solution.copy()
    solution_gaussian[:, 0] = V_gaussian
    plot_hodgkin_huxley_results(t, V_gaussian, m, h, n, I_applied, I_ext, 
                              title="Hodgkin-Huxley with Gaussian Noise")
    
    # Amplitude modulation version
    solution_amplitude_mod = solution.copy()
    solution_amplitude_mod[:, 0] = V_amplitude_mod
    plot_hodgkin_huxley_results(t, V_amplitude_mod, m, h, n, I_applied, I_ext,
                              title="Hodgkin-Huxley with Amplitude Modulation")
    
    # Ocular artifact version  
    solution_ocular = solution.copy()
    solution_ocular[:, 0] = V_ocular
    plot_hodgkin_huxley_results(t, V_ocular, m, h, n, I_applied, I_ext,
                              title="Hodgkin-Huxley with Ocular Artifact")
    
    # Print noise statistics
    print("\nNoise Statistics:")
    print(f"Original signal range: {np.min(V):.2f} to {np.max(V):.2f} mV")
    print(f"Gaussian noise - Range: {np.min(V_gaussian):.2f} to {np.max(V_gaussian):.2f} mV")
    print(f"Amplitude mod - Range: {np.min(V_amplitude_mod):.2f} to {np.max(V_amplitude_mod):.2f} mV")
    print(f"Ocular artifact - Range: {np.min(V_ocular):.2f} to {np.max(V_ocular):.2f} mV")
    print(f"Combined noise - Range: {np.min(V_combined):.2f} to {np.max(V_combined):.2f} mV")