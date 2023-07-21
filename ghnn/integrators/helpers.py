from ghnn.integrators.explicit import euler
from ghnn.integrators.symplectic import symp_euler_sep, leap_frog_sep, stoermer_verlet_sep, symp_euler, leap_frog, stoermer_verlet, impl_midpoint

__all__ = ['integrator_from_name']

def integrator_from_name(name, expl=False):
    """Identifies the integrator from a given string

    Args:
        name (str): Name of the integrator.
        expl (bool): Whether the explicit version should be used.

    Returns:
        integrator: The correct integrator.
    """
    if name == 'Euler':
        if expl:
            integrator = euler
        else:
            raise NotImplementedError
    elif name == 'Symplectic Euler':
        if expl:
            integrator = symp_euler_sep
        else:
            integrator = symp_euler
    elif name == 'Leap Frog':
        if expl:
            integrator = leap_frog_sep
        else:
            integrator = leap_frog
    elif name == 'Stoermer Verlet':
        if expl:
            integrator = stoermer_verlet_sep
        else:
            integrator = stoermer_verlet
    elif name == 'Midpoint':
        integrator = impl_midpoint
    else:
        raise ValueError('No known integrator could be identified.')

    return integrator
