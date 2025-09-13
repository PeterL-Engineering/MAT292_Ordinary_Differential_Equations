import numpy as np
import matplotlib.pyplot as plt

def plot_polynomial(coefficients, x_range, num_points=100):
    """
    Function Description:
    Plots a polynomial function over a specified range.

    Parameters:
    coefficients (list): List of coefficients for the polynomial in descending order.
                         e.g., [1, -2, 3] represents x^2 - 2x + 3.
    x_range (tuple): A tuple (x_min, x_max) defining the plot range.
    num_points (int): The number of points to generate for the plot.

    Returns:
    None: This function displays a plot using matplotlib.
    """
    
    # Unpack the range and create the x values
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, num_points)

    # Generate the polynomial y values using numpy's polyval
    y = np.polyval(coefficients, x)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)  # 'b-' is a blue solid line
    plt.title(f"Plot of Polynomial: {np.poly1d(coefficients)}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.5)  # Add x-axis
    plt.axvline(0, color='black', linewidth=0.5)  # Add y-axis
    plt.show()

if __name__ == "__main__":
    # Plot a simple parabola: x^2 - 2x - 3
    plot_polynomial(coefficients=[1, -2, -3], x_range=(-3, 5))