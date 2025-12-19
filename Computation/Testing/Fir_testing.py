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
from Numerical_Methods.Runge_Kutta import runge_kutta
from Signal_Processing.Noise_Generator import apply_all_noise_types
from Signal_Processing.Signal_Filter import fir_filter, hex_to_decimal, spike_detection

if __name__ == "__main__":
    t_span = (0, 100)  # ms
    n_steps = 2000
    
    # Initial conditions
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
    initial_condition = y0_k_activated
    np.random.seed(42)
    
    # Solve the Hodgkin-Huxley equations
    t, solution = runge_kutta(hh_ode, t_span, initial_condition, n_steps, I_ext)

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
    gaussian_std = 90
    modTaper = 100 
    modWidth = 500   
    modMinRelAmplitude = 0.5

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
    
    rms_error_noise = np.sqrt(np.mean((V-V_combined)**2))
    rms_error_filter = np.sqrt(np.mean((V[1:] - V_filtered)**2))
    print("\nRMS Noise Error:", rms_error_noise)
    print("\nRMS Filter Error:", rms_error_filter)

    spike_times, threshold = spike_detection(V_filtered)
    print("\nSpike times:", spike_times)

    # Universal font size parameters
    TITLE_FONTSIZE = 24
    AXIS_FONTSIZE = 18
    LEGEND_FONTSIZE = 16
    TICK_FONTSIZE = 16
    
    # Plot filtered signal with noisy signal and original signal
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Plot 1
    axes[0].plot(t, V, 'b-', linewidth=1.5)
    axes[0].set_title('Original Hodgkin-Huxley Membrane Potential', fontsize=TITLE_FONTSIZE)
    axes[0].set_ylabel('V (mV)', fontsize=AXIS_FONTSIZE)
    axes[0].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(['Original'], loc='upper right', fontsize=LEGEND_FONTSIZE)

    # Plot 2
    axes[1].plot(t, V_combined, 'r-', linewidth=1.5)
    axes[1].set_title('Combined Noise Effects', fontsize=TITLE_FONTSIZE)
    axes[1].set_ylabel('V (mV)', fontsize=AXIS_FONTSIZE)
    axes[1].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(['Combined Noise'], loc='upper right', fontsize=LEGEND_FONTSIZE)

    # Plot 3
    axes[2].plot(t[:-1], V_filtered, 'g-', linewidth=1.5)
    axes[2].set_title('FIR Filtered Signal', fontsize=TITLE_FONTSIZE)
    axes[2].set_xlabel('Time (ms)', fontsize=AXIS_FONTSIZE)
    axes[2].set_ylabel('V (mV)', fontsize=AXIS_FONTSIZE)
    axes[2].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(['Filtered Signal'], loc='upper right', fontsize=LEGEND_FONTSIZE)

    # Add extra space between plots 2 and 3
    plt.subplots_adjust(hspace=0.6)

    plt.tight_layout()
    plt.show()

    # print("Spike times:", spike_times)
    # print("ISI:", np.diff(spike_times))

    # fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # axes[0].eventplot(spike_times, lineoffsets=1, linelengths=0.8, colors='k')
    # axes[0].set_title('Spike Raster Plot', fontsize=24)
    # axes[0].set_xlabel('Time (ms)', fontsize=18)
    # axes[0].set_yticks([])
    # axes[0].set_xlim(t[0], t[-1])
    
    # spike_intervals = np.diff(spike_times)  # ISI in ms

    # axes[1].hist(spike_intervals, bins=20, color='c', edgecolor='k')
    # axes[1].set_title('Inter-Spike Interval (ISI) Histogram', fontsize=24)
    # axes[1].set_xlabel('Interval (ms)', fontsize=18)
    # axes[1].set_ylabel('Count', fontsize=18)
    # axes[1].grid(True, alpha=0.3)
    # plt.subplots_adjust(hspace=0.6)
    # plt.show()