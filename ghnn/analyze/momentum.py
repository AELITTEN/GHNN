"""Functions to calculate momentum in an N body problem."""
import os
from itertools import product
import numpy as np

__all__ = ['angular_momentum']

def angular_momentum(data, bodies, dimensions, m):
    """Caluclates the angular momentum of N bodies for a whole tajectory.

    Args:
        data (pd.DataFrame): Positional and momentum data to calulate the angular momentum.
        bodies (str[]): Identifiers of bodies in the data.
        dimensions (str[]): Dimensions in the data.
        m (float[]): Masses of the bodies.
    Returns:
        np.ndarray: The angular momentum of a trajectory.
    """
    N = len(bodies)
    dims = len(dimensions)
    q = data[['q_'+body+'_'+dim for (body, dim) in product(bodies, dimensions)]].values
    q = np.reshape(q, (q.shape[0], N, dims))
    p = data[['p_'+body+'_'+dim for (body, dim) in product(bodies, dimensions)]].values
    p = np.reshape(p, (p.shape[0], N, dims))

    ang_mom = np.cross(p,q)
    if dims == 2:
        ang_mom = np.sum(ang_mom, axis=-1)
        ang_mom = np.abs(ang_mom)
    elif dims == 3:
        ang_mom = np.sum(ang_mom, axis=-2)
        ang_mom = np.linalg.norm(ang_mom, axis=-1)
    else:
        raise ValueError
    return ang_mom
