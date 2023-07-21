"""Some data helpers."""
import numpy as np

__all__ = ['rotate2d']

def rotate2d(p, theta):
    """Rotates 2D vector by given angle.

    Args:
        p (np.ndarray): Vector(s) to be rotated (last dimension must be 2).
        theta (float): Angle of rotation in rad.

    Returns:
        np.ndarray: Array with elements in last dim rotated by theta.
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],[s, c]])
    R = np.transpose(R)
    return p.dot(R)
