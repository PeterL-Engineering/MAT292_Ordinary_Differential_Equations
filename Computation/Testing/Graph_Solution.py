import matplotlib.pyplot as plt

def plot_hodgkin_huxley_results(t, V, m, h, n, I_applied, I_ext=None, title=None, figsize=(12, 10)):
    """
    Plot Hodgkin-Huxley model simulation results.
    """
    # Universal font size parameters
    TITLE_FONTSIZE = 24
    AXIS_FONTSIZE = 18
    LEGEND_FONTSIZE = 16
    TICK_FONTSIZE = 16
    
    # Create figure and subplots explicitly
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Generate title
    if title is None and I_ext is not None:
        fig.suptitle(f'Hodgkin-Huxley Model - {I_ext.__name__[6:].replace("_", " ").title()} Current', 
                     fontsize=TITLE_FONTSIZE)
    elif title is None:
        fig.suptitle('Hodgkin-Huxley Model', fontsize=TITLE_FONTSIZE)
    else:
        fig.suptitle(title, fontsize=TITLE_FONTSIZE)

    # Plot 1: Membrane Potential
    ax1.plot(t, V, 'b-', linewidth=2)
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=AXIS_FONTSIZE)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax1.legend(['V(t)'], loc='upper right', fontsize=LEGEND_FONTSIZE)

    # Plot 2: Gating Variables
    ax2.plot(t, m, 'r-', label='m (Na activation)', alpha=0.7)
    ax2.plot(t, h, 'g-', label='h (Na inactivation)', alpha=0.7)
    ax2.plot(t, n, 'b-', label='n (K activation)', alpha=0.7)
    ax2.set_ylabel('Gating Variables', fontsize=AXIS_FONTSIZE)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax2.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)

    # Plot 3: Applied Current
    ax3.plot(t, I_applied, 'k-', linewidth=2)
    ax3.set_xlabel('Time (ms)', fontsize=AXIS_FONTSIZE)
    ax3.set_ylabel('Applied Current (μA/cm²)', fontsize=AXIS_FONTSIZE)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax3.legend(['I_ext(t)'], loc='upper right', fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    plt.show()