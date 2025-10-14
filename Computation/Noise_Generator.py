import numpy as np

def gaussian_noise(input_signal, mean=0.0, std_dev=1.0, seed=None):
    """
    Adds Gaussian noise to input signal.
    
    Parameters:
        - input_signal (np.ndarray): input signal array
        - mean (float): mean of Gaussian noise (default 0.0)
        - std_dev (float): standard deviation of Gaussian noise (default 1.0)
        - seed (int): random seed for reproducible results (optional)
    
    Returns:
        - np.ndarray: signal with added Gaussian noise
    """
    if std_dev < 0:
        raise ValueError("Standard deviation must be non-negative")
    
    # Use seeded random generator if seed provided
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.normal(mean, std_dev, input_signal.shape)
    else:
        noise = np.random.normal(mean, std_dev, input_signal.shape)
    
    return input_signal + noise

import numpy as np

def amplitude_modulation(input_signal, modTaper, modWidth, modMinRelAmplitude, dt=0.1):
    """
    Adds amplitude modulation to input signal
    
    Parameters:
        - input_signal (np.ndarray): input signal array
        - modTaper (int): number of samples over which burst modulation tapers
        - modWidth (int): total length of modulation in samples (includes taper periods)
        - modMinRelAmplitude (float): ratio of minimum modulated amplitude to original amplitude (0-1)
        - dt (float): time step between samples (default 0.1)
        
    Returns:
        - np.ndarray: signal with added amplitude modulation
    """
    
    # Create a copy to avoid modifying original
    modulated_signal = input_signal.copy()
    n_samples = len(input_signal)
    
    # Calculate the flat modulation duration (excluding tapers)
    flat_duration = modWidth - 2 * modTaper
    if flat_duration < 0:
        raise ValueError("modWidth must be at least 2 * modTaper")
    
    # Choose random starting point ensuring modulation fits within signal
    max_start = n_samples - modWidth
    if max_start <= 0:
        raise ValueError("Signal too short for specified modulation width")
    
    start_idx = np.random.randint(0, max_start)
    
    # Create taper profiles (quadratic)
    # For up-taper: goes from modMinRelAmplitude to 1.0
    # For down-taper: goes from 1.0 to modMinRelAmplitude
    
    # Up-taper (start of modulation)
    up_taper_indices = np.arange(modTaper)
    up_taper = modMinRelAmplitude + (1 - modMinRelAmplitude) * (up_taper_indices / modTaper) ** 2
    
    # Down-taper (end of modulation)  
    down_taper_indices = np.arange(modTaper)
    down_taper = 1.0 - (1 - modMinRelAmplitude) * (down_taper_indices / modTaper) ** 2
    
    # Apply modulation
    # Start taper
    start_taper_start = start_idx
    start_taper_end = start_idx + modTaper
    modulated_signal[start_taper_start:start_taper_end] *= up_taper
    
    # Flat modulation (constant reduced amplitude)
    flat_start = start_idx + modTaper
    flat_end = start_idx + modTaper + flat_duration
    modulated_signal[flat_start:flat_end] *= modMinRelAmplitude
    
    # End taper
    end_taper_start = start_idx + modTaper + flat_duration
    end_taper_end = start_idx + modWidth
    modulated_signal[end_taper_start:end_taper_end] *= down_taper
    
    return modulated_signal

import numpy as np

def ocular_artifact(input_signal, dt=0.1):
    """
    Adds large amplitude spikes to input signal mimicking ocular artifacts
    
    Parameters:
        - input_signal (np.ndarray): input signal array
        - dt (float): time step between samples (default 0.1)       
    Returns:
        - np.ndarray: signal with added ocular artifacts
    """
    
    artifact_duration = 1.0  # seconds of artifact duration
    artifact_length = int(artifact_duration / dt)  # convert to samples
    n_samples = len(input_signal)
    max_start = n_samples - artifact_length
    
    if max_start <= 0:
        raise ValueError("Signal too short to add ocular artifact")
    
    start_idx = np.random.randint(0, max_start)
    artifact_amplitude = 1.5 * np.max(np.abs(input_signal))  # Use absolute max for amplitude
    
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