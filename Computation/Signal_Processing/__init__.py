# Signal_Processing/__init__.py
from .Noise_Generator import gaussian_noise, amplitude_modulation, ocular_artifact
from .Signal_Filter import fir_filter, hex_to_decimal, spike_detection

__all__ = [
    'gaussian_noise',
    'amplitude_modulation',
    'ocular_artifact',
    'fir_filter', 
    'hex_to_decimal',
    'spike_detection'
]