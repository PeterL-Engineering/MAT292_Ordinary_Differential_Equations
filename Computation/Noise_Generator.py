import numpy as np

def noise_generator(solution):
    """
    Function Description:
        - Adds typical noise present in EEG and neurosignal readings
        - Gaussian noise, cardiac, ocular and muscle artefacts
         
    Parameters
        - solution (array_like): Numpy array of voltage values of the solution curve to which the function adds noise
          
    Returns
        - noisy_solution (array_like): Numpy array of voltage values with added noise
    """

    gaussian_noise = np.random.normal(0, std, size)

    # Add cardiac, ocular, and muscle artefacts

def moddulation_generator(solution):
    """
    Function Description:
        - Adds modulation to solution voltage curve
        - Uses amplitude and burst modulation
         
    Parameters
        - solution (array_like): Numpy array of voltage values of the solution curve to which the function adds modulation
          
    Returns
        - modulated_solution (array_like): Numpy array of voltage values with added modulation
    """