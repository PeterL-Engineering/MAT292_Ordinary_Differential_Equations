import numpy as np
import matplotlib.pyplot as plt

def plot_rms_results(noise, filter, figsize=(8, 6)):
    # Universal font size parameters
    TITLE_FONTSIZE = 24
    AXIS_FONTSIZE = 18
    LEGEND_FONTSIZE = 16
    TICK_FONTSIZE = 16
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot as scatter points
    ax1.set_title('Noise RMS Error vs. Filtered RMS Error', fontsize=TITLE_FONTSIZE)
    ax1.scatter(noise, filter, color='blue', s=50, alpha=0.7)
    ax1.set_xlabel('Noise RMS Error', fontsize=AXIS_FONTSIZE)
    ax1.set_ylabel('Filtered RMS Error', fontsize=AXIS_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rms_noise = np.array([20.7, 12.7, 28.1, 16.5, 33.0, 23.3, 19.2, 23.0, 30.0, 25.7, 29.4, 33.4, 34.7, 36.5, 42.3, 44.1, 46.8, 46.5, 49.4, 51.7, 54.7, 58.0, 60.9, 62.8, 63.8, 65.8, 70.6, 72.6, 75.5])
    rms_filter = np.array([35.1, 32.0, 37.4, 30.3, 36.4, 32.1, 32.5, 32.3, 34.5, 32.2, 31.9, 31.7, 30.5, 30.7, 31.5, 31.6, 31.8, 32.2, 32.6, 33.9, 34.4, 33.7, 34.2, 34.5, 35.5, 36.0, 35.6, 36.0, 36.7])
    plot_rms_results(rms_noise, rms_filter)