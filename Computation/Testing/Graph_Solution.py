import matplotlib.pyplot as plt

def plot_hodgkin_huxley_results(t, V, m, h, n, I_applied, I_ext=None, title=None, figsize=(12, 10)):
    """
    Plot Hodgkin-Huxley model simulation results
    """
    # Create figure and subplots explicitly
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Generate title
    if title is None and I_ext is not None:
        fig.suptitle(f'Hodgkin-Huxley Model - {I_ext.__name__[6:].replace("_", " ").title()} Current', fontsize=14)
    elif title is None:
        fig.suptitle('Hodgkin-Huxley Model', fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)

    # Plot 1: Membrane Potential
    ax1.plot(t, V, 'b-', linewidth=2)
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['V(t)'], loc='upper right')

    # Plot 2: Gating Variables
    ax2.plot(t, m, 'r-', label='m (Na activation)', alpha=0.7)
    ax2.plot(t, h, 'g-', label='h (Na inactivation)', alpha=0.7)
    ax2.plot(t, n, 'b-', label='n (K activation)', alpha=0.7)
    ax2.set_ylabel('Gating Variables', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Plot 3: Applied Current
    ax3.plot(t, I_applied, 'k-', linewidth=2)
    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.set_ylabel('Applied Current (μA/cm²)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(['I_ext(t)'], loc='upper right')

    plt.tight_layout()
    plt.show()