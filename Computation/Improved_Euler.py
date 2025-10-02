import numpy as np

def improved_euler_method(init_conditions, time, num_timesteps, eq_matrix):
    """
    Function Description:
        - Solves a given differential equation using Improved Euler's method
        - This method uses an average of the slope at the beginning and estimated end of the interval
    
    Parameters:
        - init_conditions (list): List of initial conditions (TIME IS INDEX 0)
        - time: length of time over which to compute the solution
        - num_timesteps: number of time steps to divide the length of time
        - eq_matrix: matrix representation of the differential equation 
    
    Returns:
        - x: a vector containing the time points as determined by time + dt
        - y: a vector containing the approximated solutions with dimensions 1 x (time / num_timesteps)
    """
    
    # Set values from initial conditions list parameter
    t_0 = init_conditions[0]
    y_0 = init_conditions[1]

    # Compute the finite time step change dt
    dt = time / num_timesteps
    t = np.arange(t_0, t_0 + time + dt, dt)  

    # Initialize the solution vector "solution" that will contain the solution x(t) and y(t)
    solution = np.full((4, len(t)), np.nan) 
    solution[0, 0] = t_0  
    solution[1, 0] = y_0  

    # Iteratively approximate the solution values at each timestep using Improved Euler
    for i in range(1, len(t)):
        # Step 1: Predictor (Euler's method)
        predictor = solution[:, i-1] + dt * np.dot(eq_matrix, solution[:, i-1])
        
        # Step 2: Corrector (average of slopes)
        slope_initial = np.dot(eq_matrix, solution[:, i-1])
        slope_predictor = np.dot(eq_matrix, predictor)
        avg_slope = (slope_initial + slope_predictor) / 2
        
        # Update solution with corrected value
        solution[:, i] = solution[:, i-1] + dt * avg_slope

    # Return solution vectors x and y from vector solution
    x = solution[0, :]
    y = solution[1, :] 

    return x, y

if __name__ == "__main__":
    
    init = [0, 2]  # [t_0, y_0]
    time = 10
    timesteps = 100
    