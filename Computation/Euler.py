import numpy as np
import matplotlib.pyplot as plt

def euler_method(init_conditions, time, num_timesteps, eq_matrix):
    """
    Function Description:
        - Solves a given differential equation by calculating the tangent slope at time step t_i
    
    Parameters:
        - init_conditions (list): List of initial conditions eg. y_0 and t_0
        - time: length of time over which to compute the solution
        - num_timesteps: number of time steps to divide the length of time
        - eq_matrix: matrix representation of the differential equation 
    
    Returns:
        - y: a vector containing the approximated solutions with dimensions 1 x (time / num_timesteps)
        - x: a vector containing the time points as determined by time + dt

    """
    
    # Set values from initial conditions list parameter
    t_0 = init_conditions[0]
    y_0 = init_conditions[1]

    # Compute the finite time step change dt
    dt = time / num_timesteps
    t = np.arange(t_0, t_0 + time + dt, dt)  

    # Initialize the solution vector "solution" that will contain the solution x(t) and y(t)
    solution = np.full((2, len(t)), np.nan) 
    solution[0, 0] = t_0  # 
    solution[1, 0] = y_0  # 

    # Iteratively approximate the solution values at each timestep
    for i in range(1, len(t)):
        solution[:, i] = solution[:, i-1] + dt * np.dot(eq_matrix, solution[:, i-1])

    # Return solution vectors x and y from vector solution
    x = solution[0, :]
    y = solution[1, :] 

    return x, y


if __name__ == "__main__":
    
    # Define parameters to send to euler method solver
    init = [0, 2] # [t_0, y_0]
    time = 60
    timesteps = 400
    eq_matrix = np.array 