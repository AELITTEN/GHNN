"""A bunch of explicit integrators."""

__all__ = ['euler']

def euler(p, q, h, grad_p, grad_q):
    """The Euler method."""
    p_new = p - h * grad_q(p, q)
    q_new = q + h * grad_p(p, q)
    return p_new, q_new
