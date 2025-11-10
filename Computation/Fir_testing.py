import numpy as np
import matplotlib.pyplot as plt
from HH_ODE import hh_ode
from Applied_Current import I_ext_burst, I_ext_double_pulse, I_ext_fm, I_ext_noisy, I_ext_ramp, I_ext_sinusoidal
from Graph_Solution import plot_hodgkin_huxley_results
from Improved_Euler import improved_euler_method
from Noise_Generator import gaussian_noise, amplitude_modulation, ocular_artifact
from Signal_Filter import fir_filter, hex_to_decimal


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

    #Apply filtering to noisy signal
    hex_coeff = ["0xFFCE", "0xFF4A", "0xFE3A", "0xFCA6", "0xFA9C", "0xF82B", "0xF567", "0xF267", "0xEF45", "0xEC1D", "0xE90F", "0xE63D", "0xE3C9", "0xE1D5", "0xE083", "0xE0F8", "0xE355", "0xE6B7", "0xEB1A", "0xF077", "0xF6C3", "0xFDF0", "0x05EB", "0x0E9D", "0x17EC", "0x21BA", "0x2BE6", "0x364D", "0x40C7", "0x4B2D", "0x5557", "0x5F1F"]
    coeffs = hex_to_decimal(hex_coeff)
    V_filtered = fir_filter(coeffs, V_combined)
    
    #plot filtered signal with noisy signal and original signal
     # Create a comprehensive comparison plot
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Original signal
    plt.subplot(5, 1, 1)
    plt.plot(t, V, 'b-', linewidth=1.5)
    plt.title('Original Hodgkin-Huxley Membrane Potential', fontsize=12)
    plt.ylabel('V (mV)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Original'], loc='upper right')
    
    # Plot 2: Combined noise
    plt.subplot(5, 1, 2)
    plt.plot(t, V_combined, 'r-', linewidth=1.5)
    plt.title('Combined Noise Effects', fontsize=12)
    plt.ylabel('V (mV)')
    plt.xlabel('Time (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Combined Noise'], loc='upper right')

    # Plot 3: Filtered Signal
    plt.subplot(5, 1, 3)
    plt.plot(t, V_filtered, 'g-', linewidth=1.5)
    plt.title('FIR Filtered Signal', fontsize=12)
    plt.ylabel('V (mV)')
    plt.xlabel('Time (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Filtered Signal'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

    # Also plot filtered version with full Hodgkin-Huxley plot
    print("\nGenerating detailed plot for filtered signal...")
    
    # filtered version
    solution_filtered = solution.copy()
    solution_filtered[:, 0] = V_filtered
    plot_hodgkin_huxley_results(t, V_filtered, m, h, n, I_applied, I_ext, 
                              title="Hodgkin-Huxley with Filtered Signal")