import numpy as np

def fir_filter(coeffs, data_in):
    """
    Function Description:
        - Implements a Finite Impulse Response (FIR) filter using circular buffer
        - Computes convolution between filter coefficients and input data
    
    Parameters:
        - coeffs (array_like): FIR filter coefficients array
        - data_in (array_like): Input signal data to be filtered
    
    Returns:
        - y (list): Filtered output signal
    """
    x_arr = np.zeros(len(coeffs)) 
    y = []  # initialize output array

    for i in range(len(data_in) - 1):  
        circ_ptr = i % len(coeffs)  # pointer for circular buffer
        x_arr[circ_ptr] = data_in[i]
        y.append(np.sum(np.multiply(x_arr, coeffs)))

    return y

def hex_to_decimal(hex_array, bit_width=16):
    """
    Function Description:
        - Converts an array of hexadecimal strings to decimal integers
        - Properly handles signed integers (two's complement representation)
    
    Parameters:
        - hex_array (list): Array of hexadecimal strings (e.g., ["0xFFCE", "0xFF4A"])
        - bit_width (int): Bit width of the numbers (default 16 for 16-bit signed)
    
    Returns:
        - dec_array (list): Array of decimal integer values
    """
    dec_array = []
    max_unsigned = 1 << bit_width  # 2^bit_width
    sign_threshold = 1 << (bit_width - 1)  # 2^(bit_width-1)
    
    for hex_str in hex_array:
        # Convert to integer
        int_val = int(hex_str, 16)
        
        # Handle signed integers
        if int_val >= sign_threshold:
            int_val -= max_unsigned
        
        dec_array.append(int_val)
    
    return dec_array

def spike_detection(voltage_data, k=4.0, sampling_rate=20000, refract=2):
    """
    Function Description:
        - Detects spikes in voltage data using percentile-based thresholding
        - Implements refractory period to prevent duplicate spike detection
    
    Parameters:
        - voltage_data (array_like): Voltage signal data for spike detection
        - k (float): Multiplier for noise standard deviation (default: 4.0)
        - sampling_rate (int): Sampling frequency in Hz (default: 20000)
        - refract (int): Refractory period in milliseconds (default: 2)
    
    Returns:
        - spike_times (list): List of spike times in milliseconds
        - threshold (float): Calculated detection threshold value
    """
    voltage = np.array(voltage_data).flatten()
    
    print(f"Voltage range: {voltage.min():.2f} to {voltage.max():.2f} mV")
    
    # For spike detection, we need a negative threshold since spikes go upward from negative baseline
    # Calculate noise from the baseline (negative values)
    baseline_mask = voltage < np.percentile(voltage, 80)  # Use lower 80% as baseline
    baseline_noise_std = np.std(voltage[baseline_mask])
    
    # Threshold should be above baseline but below spike peaks
    threshold = np.percentile(voltage, 95)  # Use 95th percentile as threshold
    
    print(f"Baseline noise std: {baseline_noise_std:.2f} mV")
    print(f"95th percentile threshold: {threshold:.2f} mV")
    
    spike_times = []
    samples = int(sampling_rate * (refract / 1000))
    last_spike_time = -np.inf
    
    for i in range(1, len(voltage)):
        if (voltage[i-1] < threshold) and (voltage[i] >= threshold):
            if (i - last_spike_time) > samples:
                spike_time_ms = (i / sampling_rate) * 1000
                spike_times.append(spike_time_ms)
                last_spike_time = i
    
    return spike_times, threshold