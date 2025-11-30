
# Neural Signal Modelling Using Hodgkin-Huxley Equations

MAT292 Project: A computational investigation of the Hodgkin-Huxley model for neuron dynamics, featuring multiple numerical solvers and signal processing with synthetic noise.

## Project Overview

This project implements the classic Hodgkin-Huxley (HH) model to simulate the electrical activity of a neuron. The codebase is organized to provide:
- Multiple numerical methods for solving the HH ordinary differential equations (ODEs)
- Tools for generating and adding synthetic noise and artefacts to the simulated signals
- Signal filtering capabilities for noise reduction
- A standardized structure for easy testing and extensibility

## Repository Structure

```
.
├── Numerical_Methods/         # Contains ODE solvers and HH model definitions
│   ├── HH_ODE.py              # Defines the Hodgkin-Huxley ODE system
│   ├── Applied_Current.py     # Library of external current functions (I_ext)
│   ├── Euler.py               # Forward Euler method implementation
│   ├── Improved_Euler.py      # Improved Euler method implementation
│   └── Runge_Kutta.py         # 4th-order Runge-Kutta method implementation
├── Signal_Processing/         # Tools for modifying and analyzing signals
│   ├── Noise_Generator.py     # Adds noise and artefacts to simulated signals
│   └── Signal_Filter.py       # Filtering tools for noise reduction
├── Testing/                   # Unit tests and validation scripts
├── Template.py                # Standardized template for function docstrings
└── README.md                  # This file
```

## Dependencies

Ensure you have the following Python packages installed:

- `numpy`
- `matplotlib`
- `os` (standard library)
- `sys` (standard library)

You can install the required packages using pip:
```bash
pip install numpy matplotlib
```

## Usage

### Running a Basic Simulation

To run a simulation with a specific numerical method and initial conditions:

1. **Navigate** to the `Numerical_Methods` directory
2. **Open** one of the solver files (e.g., `Euler.py`, `Improved_Euler.py`, `Runge_Kutta.py`)
3. **Modify** the initial conditions and parameters in the `if __name__ == '__main__':` section:
    - Key variables to change:
        - `I_ext`: Assign this to any of the imported applied current functions from `Applied_Current.py`
        - Choose from initial condition sets: `y0_depolarized`, `y0_hyperpolarized`, `y0_post_spike`, etc.
4. **Run** the script. A plot of the solution will be generated, showing the membrane potential and gating variables over time

### Example: Running the Runge-Kutta Solver

```python
# Inside Runge_Kutta.py's __main__ section - modify these lines:

# Select initial condition from available options:
# y0_depolarized, y0_hyperpolarized, y0_post_spike, y0_na_inactivated,
# y0_k_activated, y0_mixed, y0_alternative_rest, y0_near_threshold

# Select current pattern from Applied_Current.py:
# I_ext_constant, I_ext_step, I_ext_ramp, I_ext_sinusoidal, I_ext_burst, etc.

I_ext = I_ext_burst  # Change this to desired current function
initial_condition = y0_k_activated  # Change this to desired initial state

# Solve the Hodgkin-Huxley equations using Runge-Kutta method
t, solution = runge_kutta(hh_ode, t_span, initial_condition, n_steps, I_ext)
```

### Available Initial Conditions (in Runge_Kutta.py)

- `y0_depolarized`: Simulating recent synaptic input [-45mV]
- `y0_hyperpolarized`: Recent inhibition [-80mV] 
- `y0_post_spike`: Refractory period state [20mV]
- `y0_na_inactivated`: Sodium channel blocked (drugs/pathology)
- `y0_k_activated`: Increased potassium conductance
- `y0_mixed`: Partially activated state
- `y0_alternative_rest`: Alternative resting state
- `y0_near_threshold`: Near firing threshold

### Available Current Functions (in Applied_Current.py)

- `I_ext_constant`: Constant applied current
- `I_ext_step`: Step current injection
- `I_ext_ramp`: Ramp current
- `I_ext_sinusoidal`: Sinusoidal current
- `I_ext_burst`: Burst pattern current
- `I_ext_noisy`: Current with noise

### Adding Noise and Artefacts

To simulate more realistic neural signals with noise:

1. **Navigate** to the `Signal_Processing` directory
2. **Open** `Noise_Generator.py`
3. Modify the parameters in the main section:
    - `mod_taper`: Tapering modulation of the artefact
    - `mod_width`: Width modulation of the artefact  
    - `gaussian_std`: Standard deviation for Gaussian noise
    - `rel_amplitude`: Relative amplitude of applied artefacts
4. **Run** the script to generate and plot the noisy signal

### Signal Filtering

To apply filters and analyze neural signals:

1. **Use** `Signal_Filter.py` functions to process noisy Hodgkin-Huxley data
2. Available functions include FIR filtering, hex coefficient conversion, and spike detection

## Key Files Description

- `Numerical_Methods/HH_ODE.py`: Contains the function `hh_ode(t, y, I_ext)` that defines the HH ODE system. This is used by all solvers.
- `Numerical_Methods/Applied_Current.py`: A collection of functions that define various input stimuli `I_ext(t)`.
- `Numerical_Methods/Euler.py`: Forward Euler method implementation.
- `Numerical_Methods/Improved_Euler.py`: Improved Euler (Heun's) method implementation.
- `Numerical_Methods/Runge_Kutta.py`: 4th-order Runge-Kutta method implementation.
- `Signal_Processing/Noise_Generator.py`: Adds realistic noise and artefacts to clean HH signals.
- `Signal_Processing/Signal_Filter.py`: Provides filtering functions for noise reduction.
- `Template.py`: Provides coding standards and docstring templates for consistency.

## Quick Start Example

```python
# To quickly test the Runge-Kutta solver with a step current:
# 1. Open Runge_Kutta.py
# 2. Change these lines in the __main__ section:
I_ext = I_ext_burst  # Use burst current
initial_condition = y0_resting  # Use resting state
# 3. Run the file
```