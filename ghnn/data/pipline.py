"""Data piplines to generate training data."""
import math
import os
import json
import pandas as pd
from ghnn.data import *

__all__ = ['circular_brutus', 'circular', 'pendulum', 'double_pendulum']

def circular_brutus(data_path, num_runs,
                    brutus_path='Brutus-MPI',
                    store_name='all_runs.h5.1',
                    nu=1e-1,
                    min_radius=0.9,
                    max_radius=1.2,
                    T=1e7,
                    dt=1e4,
                    seed=None):
    """Pipline to generate training data of almost circular trajectories using Brutus.

    Args:
        data_path (str, path-like object): Path to where the data is supposed to be saved.
        num_runs (int): Number of trajectories/runs.
        brutus_path (str, path-like object): Path to the brutus-mpi installation.
        store_name (str, path-like object): Name of the HDF5 store for all data.
        nu (float): Random factor for the orbital elements.
        min_radius(float): Mininal distance of the three bodies to (0,0).
        max_radius (float): Maximal distance of the three bodies to (0,0).
        T (float): End time for all trajectories.
        dt (float): Time step size.
        seed (int): Random seed for np.random.uniform.
    """
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    generate_circular_brutus_inputs(data_path, num_runs, nu=nu, min_radius=min_radius, max_radius=max_radius, seed=seed)

    run_brutus(data_path, brutus_path, num_runs, T=T, dt=dt, verb=1)

    extract_from_brutus(data_path, store_name, num_runs, scale='viralnbody', save_dims=['x', 'y'])
    combine(data_path, store_name, num_runs)

def circular(data_path, num_runs,
             store_name='all_runs.h5.1',
             nu=1e-1,
             min_radius=0.9,
             max_radius=1.2,
             T=10,
             dt=1e-2,
             integrator=('Symplectic Euler', True),
             converge=None,
             seed=None):
    """Pipline to generate training data of almost circular trajectories.

    Args:
        data_path (str, path-like object): Path to where the data is supposed to be saved.
        num_runs (int): Number of trajectories/runs.
        store_name (str, path-like object): Name of the HDF5 store for all data.
        nu (float): Random factor for the orbital elements.
        min_radius(float): Mininal distance of the three bodies to (0,0).
        max_radius (float): Maximal distance of the three bodies to (0,0).
        T (float): End time for all trajectories.
        dt (float): Time step size.
        integrator (str tuple, str, integrator method): Integrator that should be used.
        converge ((float, int)-tuple): Tolerance and max steps between to steps with size dt.
        seed (int): Random seed for np.random.uniform.
    """
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    generate_circular_inputs(data_path, num_runs, store_name, nu=nu, min_radius=min_radius, max_radius=max_radius, seed=seed)

    for i in range(num_runs):
        integrate_n_body(data_path, store_name, i, integrator, T, dt, converge=converge)

    combine(data_path, store_name, num_runs)

def pendulum(data_path, num_runs,
             store_name='all_runs.h5.1',
             nu_q=1,
             nu_p=0,
             T=10,
             dt=1e-2,
             integrator=('Symplectic Euler', True),
             converge=None,
             seed=None):
    """Pipline to generate training data of a single pendulum.

    Args:
        data_path (str, path-like object): Path to where the data is supposed to be saved.
        num_runs (int): Number of trajectories/runs.
        store_name (str, path-like object): Name of the HDF5 store for all data.
        nu_q (float): Random factor for the initial postions.
        nu_p (float): Random factor for the initial momenta.
        T (float): End time for all trajectories.
        dt (float): Time step size.
        integrator (str tuple, str, integrator method): Integrator that should be used.
        converge ((float, int)-tuple): Tolerance and max steps between to steps with size dt.
        seed (int): Random seed for np.random.uniform.
    """
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    generate_pendulum_inputs(data_path, num_runs, store_name, nu_q=nu_q, nu_p=nu_p, seed=seed)

    for i in range(num_runs):
        integrate_pendulum(data_path, store_name, i, integrator, T, dt, converge=converge)

    combine(data_path, store_name, num_runs)

def double_pendulum(data_path, num_runs,
                    store_name='all_runs.h5.1',
                    nu_q=1,
                    nu_p=0,
                    T=10,
                    dt=1e-2,
                    integrator=('Symplectic Euler', False),
                    converge=None,
                    seed=None):
    """Pipline to generate training data of a double pendulum.

    Args:
        data_path (str, path-like object): Path to where the data is supposed to be saved.
        num_runs (int): Number of trajectories/runs.
        store_name (str, path-like object): Name of the HDF5 store for all data.
        nu_q (float): Random factor for the initial postions.
        nu_p (float): Random factor for the initial momenta.
        T (float): End time for all trajectories.
        dt (float): Time step size.
        integrator (str tuple, str, integrator method): Integrator that should be used.
        converge ((float, int)-tuple): Tolerance and max steps between to steps with size dt.
        seed (int): Random seed for np.random.uniform.
    """
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    generate_double_pendulum_inputs(data_path, num_runs, store_name, nu_q=nu_q, nu_p=nu_p, seed=seed)

    for i in range(num_runs):
        integrate_pendulum(data_path, store_name, i, integrator, T, dt, converge=converge)

    combine(data_path, store_name, num_runs)
