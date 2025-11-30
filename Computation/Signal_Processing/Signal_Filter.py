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


def hex_to_decimal(hex_array):
    """
    Function Description:
        - Converts an array of hexadecimal strings to decimal integers
        - Processes each hex value using Python's int conversion
    
    Parameters:
        - hex_array (list): Array of hexadecimal strings (e.g., ["0xFFCE", "0xFF4A"])
    
    Returns:
        - dec_array (list): Array of decimal integer values
    """
    dec_array = [int(h, 16) for h in hex_array]
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
    
    # Remove the automatic scaling since your voltages are now correct
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


if __name__ == "__main__":
    hex_coeff = ["0xFFCE", "0xFF4A", "0xFE3A", "0xFCA6", "0xFA9C", "0xF82B", "0xF567", "0xF267", "0xEF45", "0xEC1D", "0xE90F", "0xE63D", "0xE3C9", "0xE1D5", "0xE083", "0xE0F8", "0xE355", "0xE6B7", "0xEB1A", "0xF077", "0xF6C3", "0xFDF0", "0x05EB", "0x0E9D", "0x17EC", "0x21BA", "0x2BE6", "0x364D", "0x40C7", "0x4B2D", "0x5557", "0x5F1F"]
    coeffs = hex_to_decimal(hex_coeff)

    filtered_data = fir_filter(coeffs, data_in)

    spike_detection(filtered_data)