"""A few functions to generate data.

For the N-body problem the arbitrary precision code Brutus can be used.
for the pendulum and double pendulum another integrator is used.

Example:
    >>> data_path = '<path-to-data>'
    >>> num_runs = 100
    >>> generate_circular_brutus_inputs(data_path, num_runs)
    >>> brutus_path = '<path-to-brutus>'
    >>> run_brutus(data_path, brutus_path, num_runs, verb=1)
"""
import os
import math
from string import ascii_lowercase, ascii_uppercase
import json
from itertools import product
import subprocess
import numpy as np
import pandas as pd
from ghnn.data.helpers import rotate2d
from ghnn.integrators.helpers import integrator_from_name
from ghnn.gradients.pendulum import *
from ghnn.gradients.n_body import *
from ghnn.constants import MSun, MEarth, au, G, trappist_system

__all__ = ['generate_circular_brutus_inputs', 'generate_circular_inputs', 'generate_pendulum_inputs', 'generate_double_pendulum_inputs', 'run_brutus', 'integrate_n_body', 'integrate_pendulum']

def generate_circular_brutus_inputs(data_path, num_runs, nu=2e-1, min_radius=0.9, max_radius=1.2, seed=None):
    """Generates brutus input snapshots with 3 bodies in 2D and almost circuar trajectories.

    Three bodies are positioned on a circle with a random radius in [min_radius,max_radius].
    All bodies with a distance of 120° from one another. The velocities of the thre bodies are chosen
    such that they all stay on a circular orbit around (0,0). Then the velocities are finally disturbed
    by multiplying them each with an independent random number from [-nu,nu].

    Args:
        data_path (str, path-like object): Path to where the brutus data
          is suposed to be saved in a subdir called ‘brutus_runs’.
        num_runs (int): Number of runs for which input snapshots are supposed to be created.
        nu(float): Maximal value of the random factor for the velocities.
        min_radius(float): Mininal distance of the three bodies to (0,0).
        max_radius (float): Maximal distance of the three bodies to (0,0).
        seed (int): Random seed for np.random.uniform.
    """
    if seed != None:
        np.random.seed(seed)
    q1 = 2*np.random.rand(num_runs, 2) - 1
    r = np.random.rand(num_runs) * (max_radius-min_radius) + min_radius

    ratio = r/np.sqrt(np.sum((q1**2), axis=1))
    q1 *= np.tile(np.expand_dims(ratio, 1), (1, 2))
    q2 = rotate2d(q1, theta=2*np.pi/3)
    q3 = rotate2d(q2, theta=2*np.pi/3)

    # velocity that yields a circular orbit
    v1 = rotate2d(q1, theta=np.pi/2)
    v1 = v1 / np.tile(np.expand_dims(r**1.5, axis=1), (1, 2))
    v1 = v1 * np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))
    v2 = rotate2d(v1, theta=2*np.pi/3)
    v3 = rotate2d(v2, theta=2*np.pi/3)

    # make the circular orbits slightly chaotic
    v1 *= 1 + nu*(2*np.random.rand(2) - 1)
    v2 *= 1 + nu*(2*np.random.rand(2) - 1)
    v3 *= 1 + nu*(2*np.random.rand(2) - 1)

    scales = {'mass': 1, 'position': 1, 'velocity': 1, 'time': 1}

    if not os.path.exists(os.path.join(data_path, 'brutus_runs')):
        os.mkdir(os.path.join(data_path, 'brutus_runs'))

    for i in range(num_runs):
        with open(os.path.join(data_path, 'brutus_runs', f'run{i}.dat'), 'w') as file_:
            file_.write('0 3 0\n')
            file_.write(f'1.0 {q1[i][0]:21.18f} {q1[i][1]:21.18f} 0.0 {v1[i][0]:21.18f} {v1[i][1]:21.18f} 0.0\n')
            file_.write(f'1.0 {q2[i][0]:21.18f} {q2[i][1]:21.18f} 0.0 {v2[i][0]:21.18f} {v2[i][1]:21.18f} 0.0\n')
            file_.write(f'1.0 {q3[i][0]:21.18f} {q3[i][1]:21.18f} 0.0 {v3[i][0]:21.18f} {v3[i][1]:21.18f} 0.0\n')
        with open(os.path.join(data_path, 'brutus_runs', f'scales{i}.json'), 'w') as file_:
            json.dump(scales, file_)

def generate_circular_inputs(data_path, num_runs, store_name, nu=2e-1, min_radius=0.9, max_radius=1.2, seed=None):
    """Generates input coordinates with 3 bodies in 2D and almost circuar trajectories.

    Three bodies are positioned on a circle with a random radius in [min_radius,max_radius].
    All bodies with a distance of 120° from one another. The velocities of the thre bodies are chosen
    such that they all stay on a circular orbit around (0,0). Then the velocities are finally disturbed
    by multiplying them each with an independent random number from [-nu,nu].

    Args:
        data_path (str, path-like object): Path to where the data is supposed to be saved.
        num_runs (int): Number of trajectories/runs.
        store_name (str, path-like object): Name of the HDF5 store for all data.
        nu(float): Maximal value of the random factor for the velocities.
        min_radius(float): Mininal distance of the three bodies to (0,0).
        max_radius (float): Maximal distance of the three bodies to (0,0).
        seed (int): Random seed for np.random.uniform.
    """
    if seed != None:
        np.random.seed(seed)

    bodies = ['A', 'B', 'C']
    dims = ['x', 'y']
    m = [[1., 1., 1.]] * num_runs
    scale = 'viralnbody'

    q1 = 2*np.random.rand(num_runs, 2) - 1
    r = np.random.rand(num_runs) * (max_radius-min_radius) + min_radius

    ratio = r/np.sqrt(np.sum((q1**2), axis=1))
    q1 *= np.tile(np.expand_dims(ratio, 1), (1, 2))
    q2 = rotate2d(q1, theta=2*np.pi/3)
    q3 = rotate2d(q2, theta=2*np.pi/3)

    # velocity that yields a circular orbit
    v1 = rotate2d(q1, theta=np.pi/2)
    v1 = v1 / np.tile(np.expand_dims(r**1.5, axis=1), (1, 2))
    v1 = v1 * np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))
    v2 = rotate2d(v1, theta=2*np.pi/3)
    v3 = rotate2d(v2, theta=2*np.pi/3)

    # make the circular orbits slightly chaotic
    v1 *= 1 + nu*(2*np.random.rand(2) - 1)
    v2 *= 1 + nu*(2*np.random.rand(2) - 1)
    v3 *= 1 + nu*(2*np.random.rand(2) - 1)

    kwargs = {'complib': 'zlib', 'complevel': 1}
    columns = [qp+'_'+body+'_'+dim for (qp,body,dim) in product(['q', 'p'], bodies, dims)]
    for i in range(num_runs):
        p = np.concatenate((v1[i], v2[i], v3[i])) * np.repeat(m[i], len(dims))
        q = np.concatenate((q1[i], q2[i], q3[i]))
        init = pd.DataFrame([np.concatenate((q,p))], columns=columns)
        init.to_hdf(os.path.join(data_path, store_name), key='/run' + str(i), format='fixed', **kwargs)

    constants = pd.Series([bodies, dims, m, scale],
                          index=['bodies', 'dimensions', 'masses', 'scale'])
    git_dir = os.path.dirname(__file__)[:-9] + '.git'
    version = subprocess.check_output(['git',
                                       '--git-dir=' + git_dir,
                                       'rev-parse', 'HEAD']).decode('ascii').strip()
    constants['code_version'] = version
    constants.to_hdf(os.path.join(data_path, store_name), key='/constants', format='fixed', **kwargs)

def generate_pendulum_inputs(data_path, num_runs, store_name, nu_q=1, nu_p=0, seed=None):
    """Generates input coordinates for the single pendulum.

    Args:
        data_path (str, path-like object): Path to where the data is supposed to be saved.
        num_runs (int): Number of trajectories/runs.
        store_name (str, path-like object): Name of the HDF5 store for all data.
        nu_q (float): Random factor for the initial postions.
        nu_p (float): Random factor for the initial momenta.
        seed (int): Random seed for np.random.uniform.
    """
    if seed != None:
        np.random.seed(seed)

    bodies = ['A']
    p_range = math.sqrt(2)

    kwargs = {'complib': 'zlib', 'complevel': 1}
    for i in range(num_runs):
        p = nu_p * np.random.uniform(-p_range, p_range, (1))
        q = nu_q * np.random.uniform(-math.pi, math.pi, (1))
        columns = [qp+'_'+body for (qp, body) in product(['q', 'p'], bodies)]
        init = pd.DataFrame([np.concatenate((q,p))], columns=columns)
        init.to_hdf(os.path.join(data_path, store_name), key='/run' + str(i), format='fixed', **kwargs)

    constants = pd.Series([bodies, 1, 1, 1],
                          index=['bodies', 'mass', 'length', 'g'])
    git_dir = os.path.dirname(__file__)[:-9] + '.git'
    version = subprocess.check_output(['git',
                                       '--git-dir=' + git_dir,
                                       'rev-parse', 'HEAD']).decode('ascii').strip()
    constants['code_version'] = version
    constants.to_hdf(os.path.join(data_path, store_name), key='/constants', format='fixed', **kwargs)

def generate_double_pendulum_inputs(data_path, num_runs, store_name, nu_q=1, nu_p=0, seed=None):
    """Generates input coordinates for the double pendulum.

    Args:
        data_path (str, path-like object): Path to where the data is supposed to be saved.
        num_runs (int): Number of trajectories/runs.
        store_name (str, path-like object): Name of the HDF5 store for all data.
        nu_q (float): Random factor for the initial postions.
        nu_p (float): Random factor for the initial momenta.
        seed (int): Random seed for np.random.uniform.
    """
    if seed != None:
        np.random.seed(seed)

    bodies = ['A', 'B']
    p_range = math.sqrt(6/5)

    kwargs = {'complib': 'zlib', 'complevel': 1}
    for i in range(num_runs):
        p = nu_p * np.random.uniform(-p_range, p_range, (2))
        q = nu_q * np.random.uniform(-math.pi, math.pi, (2))
        columns = [qp+'_'+body for (qp, body) in product(['q', 'p'], bodies)]
        init = pd.DataFrame([np.concatenate((q,p))], columns=columns)
        init.to_hdf(os.path.join(data_path, store_name), key='/run' + str(i), format='fixed', **kwargs)

    constants = pd.Series([bodies, [1, 1], [1, 1], 1],
                          index=['bodies', 'masses', 'lengths', 'g'])
    git_dir = os.path.dirname(__file__)[:-9] + '.git'
    version = subprocess.check_output(['git',
                                       '--git-dir=' + git_dir,
                                       'rev-parse', 'HEAD']).decode('ascii').strip()
    constants['code_version'] = version
    constants.to_hdf(os.path.join(data_path, store_name), key='/constants', format='fixed', **kwargs)

def run_brutus(data_path, brutus_path, stop, start=0, n=2, T=10, dt=0.1, eta=0.1, e=1E-6, Lw=56, nmax=64, verb=2):
    """Runs brutus for stop-start input snapshots.

    Nothing but a wrapper for brutus-mpi with the possibility to iterate over many input snapshots
    and define the verbosity. IMPORTANT: everything (data and time) are in N-body units.

    Args:
        data_path (str, path-like object): Path to where the brutus data is in a subdir called ‘brutus_runs’.
        brutus_path (str, path-like object): Path to the brutus-mpi installation.
        stop (int): Upper boundary to the range of runs.
        start (int): Lower boundary to the range of runs.
        n (int): Number of CPU cores.
        T (float, float[]): End time. (possibly per run)
        dt (float, float[]): Snapshot time interval. (possibly per run)
        eta (float): Timestep parameter.
        e (float): Bulirsch-stoer tolerance parameter (1e-6, 1e-8, 1e-10, ...).
        Lw (int): Number of digits (56, 64, 72, ...).
        nmax (int): Maximum number of Bulirsch-Stoer iterationn.
        verb (int): Verbosity level (0, 1, 2).
    """
    if not isinstance(dt, list):
        dt = [dt] * (stop-start)
    if not isinstance(T, list):
        T = [T] * (stop-start)
    for i in range(start, stop):
        start_file = os.path.join(data_path, 'brutus_runs', f'run{i}.dat')
        with open(start_file, newline='') as file_:
            line = file_.readline()
        first_space = line.find(' ')
        second_space = line.find(' ', first_space+1)
        N = int(line[first_space+1:second_space])
        out_file = os.path.join(data_path, 'brutus_runs', f'run{i}')
        if verb == 0 or verb == 1:
            stdout = True
        else:
            stdout = False
        out = subprocess.run(['mpiexec', '-n', str(n), # number of processors to be used
                              os.path.join(brutus_path, 'main.exe'),
                              out_file, # output file for snapshots
                              '0', # begin time
                              str(T[i]+dt[i]/2), # end time
                              str(dt[i]), # snapshot time interval
                              str(eta), # timestep parameter
                              str(e), # bulirsch-stoer tolerance parameter (1e-6, 1e-8, 1e-10, ...)
                              str(Lw), # number of digits (56, 64, 72, ...)
                              str(nmax), # maximum number of Bulirsch-Stoer iterations
                              str(N), # number of objects in the initial condition file
                              'file', start_file],
                            capture_output=stdout,
                            text=True)
        if verb == 1:
            print(out.stderr[out.stderr.find('N'):])

def integrate_n_body(data_path, store_name, run_num, integrator, T, dt, converge=None):
    """Integrates a given run up to a final time T.

    Args:
        data_path (str, path-like object): Path to where the data should be saved.
        store_name (str, path-like object): Name of the HDF5 store where to save all runs.
        run_num (int): Number of run to integrate.
        integrator (str tuple, str, integrator method): Integrator that should be used.
        T (float): Final time until when the trajectory should be integrated.
        dt (float): Stepsize in the data.
        converge ((float, int)-tuple): Tolerance and max steps between to steps with size dt.
    """
    data = pd.read_hdf(os.path.join(data_path, store_name), '/run' + str(run_num)).iloc[0]
    constants = pd.read_hdf(os.path.join(data_path, store_name), '/constants')
    bodies = constants['bodies']
    dims = constants['dimensions']
    m = constants['masses'][run_num]
    if constants['scale'] == 'SI':
        g = G
    else:
        g = 1
    calculations = np.zeros((int(T/dt)+1, 2*len(bodies), len(dims)))

    grad_q = N_body_grad_q(m, g)
    grad_p = Kin_grad_p(m)

    p = np.array([data[['p_'+body+'_'+dim for dim in dims]] for body in bodies])
    q = np.array([data[['q_'+body+'_'+dim for dim in dims]] for body in bodies])
    calculations[0, :, :] = np.concatenate((q,p))

    if isinstance(integrator, tuple):
        integrator = integrator_from_name(integrator[0], integrator[1])
    elif isinstance(integrator, str):
        integrator = integrator_from_name(integrator)

    for i in range(1, int(T/dt)+1):
        p_old, q_old = p, q
        p, q = integrator(p_old, q_old, dt, grad_p, grad_q)
        if converge != None:
            tol, max_steps = converge
            diff = np.array([2*tol])
            j = 1
            while (tol < np.linalg.norm(diff) and 2**j <= max_steps):
                p_new, q_new = p, q
                p, q = p_old, q_old
                for k in range(2**j):
                    p, q = integrator(p, q, dt/(2**j), grad_p, grad_q)
                diff = np.concatenate((p-p_new, q-q_new))
                j += 1
                if max_steps < 2**j:
                    print('Warning: Too many steps required!!!')

        calculations[i, :, :] = np.concatenate((q,p))

    kwargs = {'complib': 'zlib', 'complevel': 1}
    columns = [qp+'_'+body+'_'+dim for (qp,body,dim) in product(['q', 'p'], bodies, dims)]
    data = pd.DataFrame(np.reshape(calculations, (calculations.shape[0], -1)), columns=columns)
    data['time'] = data.index * dt
    data.to_hdf(os.path.join(data_path, store_name), key='/run' + str(run_num), format='fixed', **kwargs)

    constants['step_size'] = dt
    constants.to_hdf(os.path.join(data_path, store_name), key='/constants', format='fixed', **kwargs)

def integrate_pendulum(data_path, store_name, run_num, integrator, T, dt, converge=None):
    """Integrates a given run up to a final time T.

    Args:
        data_path (str, path-like object): Path to where the data should be saved.
        store_name (str, path-like object): Name of the HDF5 store where to save all runs.
        run_num (int): Number of run to integrate.
        integrator (str tuple, str, integrator method): Integrator that should be used.
        T (float): Final time until when the trajectory should be integrated.
        dt (float): Stepsize in the data.
        converge ((float, int)-tuple): Tolerance and max steps between to steps with size dt.
    """
    data = pd.read_hdf(os.path.join(data_path, store_name), '/run' + str(run_num)).iloc[0]
    constants = pd.read_hdf(os.path.join(data_path, store_name), '/constants')
    bodies = constants['bodies']
    if len(bodies) == 1:
        calculations = np.zeros((int(T/dt)+1, 2))
        m = constants['mass']
        g = constants['g']
        l = constants['length']
        grad_q = Pendulum_grad_q(m, g, l)
        grad_p = Pendulum_grad_p(m, l)
    elif len(bodies) == 2:
        calculations = np.zeros((int(T/dt)+1, 4))
        m1 = constants['masses'][0]
        m2 = constants['masses'][1]
        g = constants['g']
        l1 = constants['lengths'][0]
        l2 = constants['lengths'][1]
        grad_q = Double_pendulum_grad_q(m1, m2, g, l1, l2)
        grad_p = Double_pendulum_grad_p(m1, m2, g, l1, l2)
    else:
        raise ValueError

    p = data[['p_'+body for body in bodies]].values
    q = data[['q_'+body for body in bodies]].values
    calculations[0, :] = np.concatenate((q,p))

    if isinstance(integrator, tuple):
        integrator = integrator_from_name(integrator[0], integrator[1])
    elif isinstance(integrator, str):
        integrator = integrator_from_name(integrator)

    for i in range(1, int(T/dt)+1):
        p_old, q_old = p, q
        p, q = integrator(p_old, q_old, dt, grad_p, grad_q)
        if converge != None:
            tol, max_steps = converge
            diff = np.array([2*tol])
            j = 1
            while (tol < np.linalg.norm(diff) and 2**j <= max_steps):
                p_new, q_new = p, q
                p, q = p_old, q_old
                for k in range(2**j):
                    p, q = integrator(p, q, dt/(2**j), grad_p, grad_q)
                diff = np.concatenate((p-p_new, q-q_new))
                j += 1
                if max_steps < 2**j:
                    print('Warning: Too many steps required!!!')

        calculations[i, :] = np.concatenate((q,p))

    kwargs = {'complib': 'zlib', 'complevel': 1}
    columns = [qp+'_'+body for (qp, body) in product(['q', 'p'], bodies)]
    data = pd.DataFrame(calculations, columns=columns)
    data['time'] = data.index * dt
    data.to_hdf(os.path.join(data_path, store_name), key='/run' + str(run_num), format='fixed', **kwargs)

    constants['step_size'] = dt
    constants.to_hdf(os.path.join(data_path, store_name), key='/constants', format='fixed', **kwargs)
