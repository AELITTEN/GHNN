"""A bunch of symplectic integrators."""
import numpy as np
from scipy.optimize import fsolve

__all__ = ['symp_euler_sep', 'leap_frog_sep', 'stoermer_verlet_sep', 'symp_euler', 'leap_frog', 'stoermer_verlet', 'impl_midpoint']

def symp_euler_sep(p, q, h, grad_p, grad_q):
    """The symplectic Euler method for separable Hamiltonians."""
    p = p - h * grad_q(q)
    q = q + h * grad_p(p)
    return p, q

def leap_frog_sep(p, q, h, grad_p, grad_q):
    """The leapfrog integrator for separable Hamiltonians."""
    p = p - h * grad_q(q)
    q = q + h * grad_p(p)
    return p, q

def stoermer_verlet_sep(p, q, h, grad_p, grad_q):
    """The Stoermer-Verlet integrator for separable Hamiltonians."""
    p = p - h/2 * grad_q(q)
    q = q + h * grad_p(p)
    p = p - h/2 * grad_q(q)
    return p, q

def symp_euler(p, q, h, grad_p, grad_q, **kwargs):
    """The symplectic Euler method."""
    func = lambda x : x - p + h * grad_q(x, q)
    p_new = fsolve(func, p, **kwargs)
    q_new = q + h * grad_p(p_new, q)
    return p_new, q_new

def leap_frog(p, q, h, grad_p, grad_q, **kwargs):
    """The leapfrog integrator."""
    func = lambda x : x - p + h * grad_q(x, q)
    p_new = fsolve(func, p, **kwargs)
    q_new = q + h * grad_p(p_new, q)
    return p_new, q_new

def stoermer_verlet(p, q, h, grad_p, grad_q, **kwargs):
    """The Stoermer-Verlet integrator."""
    func = lambda x : x - p + h/2 * grad_q(x, q)
    p_half_new = fsolve(func, p, **kwargs)
    func = lambda x : x - q - h/2 * (grad_p(p_half_new, q) + grad_p(p_half_new, x))
    q_new = fsolve(func, q, **kwargs)
    p_new = p_half_new - h/2 * grad_q(p_half_new, q_new)
    return p_new, q_new

def impl_midpoint(p, q, h, grad_p, grad_q, **kwargs):
    """The implicit midpoint rule."""
    def func(x):
        x_p = x[:int(len(x)/2)]
        x_q = x[int(len(x)/2):]
        zero_p = x_p - p + h/2 * (grad_q(x_p, x_q) + grad_q(p, q))
        zero_q = x_q - q - h/2 * (grad_p(x_p, x_q) + grad_p(p, q))
        return np.concatenate((zero_p, zero_q))

    x_new = fsolve(func, np.concatenate((p, q)), **kwargs)
    return x_new[:int(len(x_new)/2)], x_new[int(len(x_new)/2):]
