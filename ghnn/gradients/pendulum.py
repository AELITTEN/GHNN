"""The pendulum and double pendulum gradient classes."""
import math
import numpy as np

__all__ = ['Pendulum_grad_q', 'Pendulum_grad_p', 'Double_pendulum_grad_q', 'Double_pendulum_grad_p']

class Pendulum_grad_q:
    """Calculates forces from coordinates."""
    def __init__(self, m, g, l):
        self.m = m
        self.g = g
        self.l = l

    def __call__(self, q):
        return self.m * self.g * self.l * np.sin(q)

class Pendulum_grad_p:
    """Calculates forces from coordinates."""
    def __init__(self, m, l):
        self.m = m
        self.l = l

    def __call__(self, p):
        return p / (self.m * self.l**2)

class Double_pendulum_grad_q:
    """Calculates forces from coordinates."""
    def __init__(self, m1, m2, g, l1, l2):
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.l1 = l1
        self.l2 = l2

    def __call__(self, p, q):
        p1 = p[0]
        p2 = p[1]
        q1 = q[0]
        q2 = q[1]
        m1 = self.m1
        m2 = self.m2
        g = self.g
        l1 = self.l1
        l2 = self.l2

        C = m1 + m2 * math.sin(q1 - q2)**2
        C1 = (p1 * p2 * math.sin(q1 - q2)) / (l1 * l2 * C)
        C2 = (m2 * l2**2 * p1**2 + (m1 + m2) * l1**2 * p2**2 - 2 * m2 * l1 * l2 * p1 * p2 * math.cos(q1 - q2)) \
                / (2 * l1**2 * l2**2 * C**2) * math.sin(2 * (q1 - q2))
        grad1 = (m1 + m2) * g * l1 * math.sin(q1) + C1 - C2
        grad2 = m2 * g * l2 * math.sin(q2) - C1 + C2
        return np.array([grad1, grad2])

class Double_pendulum_grad_p:
    """Calculates forces from coordinates."""
    def __init__(self, m1, m2, g, l1, l2):
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.l1 = l1
        self.l2 = l2

    def __call__(self, p, q):
        p1 = p[0]
        p2 = p[1]
        q1 = q[0]
        q2 = q[1]
        m1 = self.m1
        m2 = self.m2
        g = self.g
        l1 = self.l1
        l2 = self.l2

        C = m1 + m2 * math.sin(q1 - q2)**2
        grad1 = (l2 * p1 - l1 * p2 * math.cos(q1 - q2)) / (l1**2 * l2 * C)
        grad2 = (l1 * (m1 + m2) * p2 - l2 * m2 * p1 * math.cos(q1 - q2)) / (l1 * l2**2 * m2 * C)
        return np.array([grad1, grad2])
