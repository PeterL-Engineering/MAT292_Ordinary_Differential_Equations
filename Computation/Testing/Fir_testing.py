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
from Signal_Processing.Noise_Generator import apply_all_noise_types
from Signal_Processing.Signal_Filter import fir_filter, hex_to_decimal, spike_detection

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
    
    # Noise parameter values
    modTaper = 100  # samples
    modWidth = 500  # samples  
    modMinRelAmplitude = 0.3  # 30% of original amplitude
    gaussian_std = 10.0

    V_gaussian, V_amplitude_mod, V_ocular, V_combined = apply_all_noise_types(
        V.copy(), dt, gaussian_std=gaussian_std, modTaper=modTaper, 
        modWidth=modWidth, modMinRelAmplitude=modMinRelAmplitude
    )

    coeffs = [
    0.000000000000000000,
    -0.000011466433343440,
    -0.000048159680438716,
    -0.000070951498745996,
    0.000000000000000000,
    0.000226394896924650,
    0.000570593070542985,
    0.000843882357908127,
    0.000742644459879189,
    -0.000000000000000001,
    -0.001387330856962112,
    -0.002974017320060805,
    -0.003876072294410999,
    -0.003078546062261788,
    0.000000000000000002,
    0.004911281747189231,
    0.009897689392343489,
    0.012239432700612913,
    0.009302643950572093,
    -0.000000000000000005,
    -0.013947257358995015,
    -0.027649479941983943,
    -0.034045985830080304,
    -0.026173578588735643,
    0.000000000000000007,
    0.043288592274982135,
    0.096612134128292462,
    0.148460098539443669,
    0.186178664190551207,
    0.199977588313552862,
    0.186178664190551235,
    0.148460098539443669,
    0.096612134128292462,
    0.043288592274982128,
    0.000000000000000007,
    -0.026173578588735653,
    -0.034045985830080325,
    -0.027649479941983936,
    -0.013947257358995019,
    -0.000000000000000005,
    0.009302643950572094,
    0.012239432700612920,
    0.009897689392343489,
    0.004911281747189235,
    0.000000000000000002,
    -0.003078546062261790,
    -0.003876072294411004,
    -0.002974017320060808,
    -0.001387330856962112,
    -0.000000000000000001,
    0.000742644459879189,
    0.000843882357908127,
    0.000570593070542984,
    0.000226394896924650,
    0.000000000000000000,
    -0.000070951498745996,
    -0.000048159680438716,
    -0.000011466433343440,
    0.000000000000000000,
]

    V_filtered = fir_filter(coeffs, V_combined)
    
    # Plot filtered signal with noisy signal and original signal
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Plot 1
    axes[0].plot(t, V, 'b-', linewidth=1.5)
    axes[0].set_title('Original Hodgkin-Huxley Membrane Potential', fontsize=12)
    axes[0].set_ylabel('V (mV)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(['Original'], loc='upper right')

    # Plot 2
    axes[1].plot(t, V_combined, 'r-', linewidth=1.5)
    axes[1].set_title('Combined Noise Effects', fontsize=12)
    axes[1].set_ylabel('V (mV)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(['Combined Noise'], loc='upper right')

    # Plot 3
    axes[2].plot(t[:-1], V_filtered, 'g-', linewidth=1.5)
    axes[2].set_title('FIR Filtered Signal', fontsize=12)
    axes[2].set_ylabel('V (mV)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(['Filtered Signal'], loc='upper right')

    # Add extra space between plots 2 and 3
    plt.subplots_adjust(hspace=0.6)

    plt.tight_layout()
    plt.show()
    
    rms_error_noise = np.sqrt(np.mean((V-V_combined)**2))
    rms_error_filter = np.sqrt(np.mean((V[1:] - V_filtered)**2))
    print("\nRMS Noise Error:", rms_error_noise)
    print("\nRMS Filter Error:", rms_error_filter)

    spike_times, threshold = spike_detection(V_filtered)
    print("\nSpike times:", spike_times)