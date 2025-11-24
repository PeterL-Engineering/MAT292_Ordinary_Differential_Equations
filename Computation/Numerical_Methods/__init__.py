# Numerical_Methods/__init__.py
from .Euler import euler_method
from .Improved_Euler import improved_euler_method
from .Runge_Kutta import runge_kutta
from .Applied_Current import I_ext_burst, I_ext_double_pulse, I_ext_fm, I_ext_noisy, I_ext_ramp, I_ext_sinusoidal
from .HH_ODE import hh_ode

__all__ = [
    'euler_method',
    'improved_euler_method', 
    'runge_kutta',
    'I_ext_burst',
    'I_ext_double_pulse',
    'I_ext_fm',
    'I_ext_noisy',
    'I_ext_ramp',
    'I_ext_sinusoidal',
    'hh_ode'
]