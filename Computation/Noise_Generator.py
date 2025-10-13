def gaussian_noise(input_signal, strength):
    """
    Function Description:
        - Adds random noise via gaussian distribution to numpy array
        
    Parameters:
        - input_signal (array_like): numpy array of signal values at given time 't'
        - strength (float): percentage of max amplitude of given input_signal for max noise added (0 - 1)
    
    Returns:
        - output_signal (array_like): numpy array of signal values with added noise
    """

    max_amplitude = max(input_signal)
    max_noise_amplitude = max_amplitude * strength

    