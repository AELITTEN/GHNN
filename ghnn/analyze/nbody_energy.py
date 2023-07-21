"""Functions to calculate energies in an N body problem."""
import os
from itertools import product
import numpy as np
from ghnn.constants import G

__all__ = ['nbody_kinetic_energy', 'nbody_potential_energy', 'nbody_all_energy']

def nbody_kinetic_energy(m, p):
    """Calulcates the kinetic energy of N bodies for one point in time or for a whole trajectory.

    Args:
        m (np.ndarray, float[]): Masses of the bodies.
        p (np.ndarray): Momenta of the bodies.

    Returns:
        float, np.ndarray: The kinetic energy at one single time or of a trajectory.
    """
    if p.ndim == 2:
        energy = 0
        for i, m_i in enumerate(m):
            energy += 0.5 * np.linalg.norm(p[i])**2 / m_i
    elif p.ndim == 3:
        energy = np.zeros(p.shape[0])
        for i, p_i in enumerate(p):
            for j, m_j in enumerate(m):
                energy[i] += 0.5 * np.linalg.norm(p_i[j])**2 / m_j
    else:
        raise ValueError()
    return energy

def nbody_potential_energy(m, q, g=G):
    """Calulcates the potential energy of N bodies for one point in time or for a whole trajectory.

    Args:
        m (np.ndarray, float[]): Masses of the bodies.
        q (np.ndarray): Positions of the bodies.
        g (float): Gravitational constant.

    Returns:
        float, np.ndarray: The potential energy at one single time or of a trajectory.
    """
    if q.ndim == 2:
        energy = 0
        for i, m_i in enumerate(m):
            for j in range(i):
                energy -= g * m_i * m[j] / np.linalg.norm(q[i] - q[j])
    elif q.ndim == 3:
        energy = np.zeros(q.shape[0])
        for i, q_i in enumerate(q):
            for j, m_j in enumerate(m):
                for k in range(j):
                    energy[i] -= g * m_j * m[k] / np.linalg.norm(q_i[j] - q_i[k])
    else:
        raise ValueError()
    return energy

def nbody_all_energy(data, bodies, dimensions, m, g=G):
    """Calulcates the total energy for a whole N body trajectory.

    Args:
        data (pd.DataFrame): Positional and momentum data to calulate the energy.
        bodies (str[]): Identifiers of bodies in the data.
        dimensions (str[]): Dimensions in the data.
        m (float[]): Masses of the bodies.
        g (float): Gravitational constant.

    Returns:
        np.ndarray: The total energy of a trajectory.
    """
    N = len(bodies)
    dims = len(dimensions)
    q = data[['q_'+body+'_'+dim for (body, dim) in product(bodies, dimensions)]].values
    q = np.reshape(q, (q.shape[0], N, dims))
    p = data[['p_'+body+'_'+dim for (body, dim) in product(bodies, dimensions)]].values
    p = np.reshape(p, (p.shape[0], N, dims))

    ke = nbody_kinetic_energy(m, p)
    pe = nbody_potential_energy(m, q, g=g)
    return ke, pe
