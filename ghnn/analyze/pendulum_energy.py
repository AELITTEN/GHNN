"""Functions to calculate energies in a singel or double pendulum."""
import os
from itertools import product
import numpy as np
from ghnn.constants import G

__all__ = ['pendulum_kinetic_energy', 'pendulum_potential_energy', 'pendulum_all_energy', 'doublependulum_all_energy']

def pendulum_kinetic_energy(m, l, p):
    """Calulcates the kinetic energy of a pendulum for one point in time or for a whole trajectory.

    Args:
        m (float): Mass of the pendulum.
        l (float): Length of the pendulum.
        p (np.ndarray): Momenta of the pendulum.

    Returns:
        float, np.ndarray: The kinetic energy at one single time or of a trajectory.
    """
    return 0.5 * p**2 / (m * l**2)

def pendulum_potential_energy(m, l, q, g=G):
    """Calulcates the potential energy of a pendulum for one point in time or for a whole trajectory.

    Args:
        m (float): Mass of the pendulum.
        l (float): Length of the pendulum.
        q (np.ndarray): Position of the pendulum.
        g (float): Gravitational constant.

    Returns:
        float, np.ndarray: The potential energy at one single time or of a trajectory.
    """
    return m * g * l * (1 - np.cos(q))

def pendulum_all_energy(data, m, l, g=G):
    """Calulcates the total energy for a whole pendulum trajectory.

    Args:
        data (pd.DataFrame): Positional and momentum data to calulate the energy.
        m (float): Mass of the pendulum.
        l (float): Length of the pendulum.
        g (float): Gravitational constant.

    Returns:
        np.ndarray: The total energy of a trajectory.
    """
    q = data[['q_A']].values
    p = data[['p_A']].values

    ke = pendulum_kinetic_energy(m, l, p)
    pe = pendulum_potential_energy(m, l, q, g=g)
    return ke, pe

def doublependulum_all_energy(data, m, l, g=G):
    """Calulcates the total energy for a double pendulum trajectory.

    Args:
        data (pd.DataFrame): Positional and momentum data to calulate the energy.
        m (float[]): Massse of the double pendulum.
        l (float[]): Lengths of the doube pendulum.
        g (float): Gravitational constant.

    Returns:
        np.ndarray: The total energy of a trajectory.
    """
    q1 = data['q_A'].values
    q2 = data['q_B'].values
    p1 = data['p_A'].values
    p2 = data['p_B'].values
    m1 = m[0]
    m2 = m[1]
    l1 = l[0]
    l2 = l[1]

    te = m2 * l2**2 * p1**2 + (m1 + m2) * l1**2 * p2**2 - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(q1 - q2)
    te /= 2 * m2 * l1**2 * l2**2 * (m1 + m2 * np.sin(q1 - q2)**2)
    te += (m1 + m2) * g * l1 * (1 - np.cos(q1)) + m2 * g * l2 * (1 - np.cos(q2)) + m1 * g * l2
    return te
