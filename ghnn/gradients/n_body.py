"""The N-body system gradient classes."""
import numpy as np

__all__ = ['N_body_grad_q', 'Kin_grad_p']

class N_body_grad_q:
    """Calculates gravitational forces from coordinates."""
    def __init__(self, m, g):
        self.m = m
        self.g = g

    def __call__(self, q):
        m = self.m
        g = self.g
        grad = np.zeros(q.shape)
        for i in range(1, q.shape[0]):
            dist = np.roll(q, i, axis=-2) - q
            grad -= g * np.expand_dims(m, axis=1) * np.expand_dims(np.roll(m, i, axis=-1) / np.linalg.norm(dist, axis=-1)**3, axis=1) * dist
        return grad

class Kin_grad_p:
    def __init__(self, m):
        self.m = m

    def __call__(self, p):
        grad = p / np.expand_dims(self.m, axis=1)
        return grad
