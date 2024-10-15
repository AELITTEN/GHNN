import os
from itertools import product
import numpy as np
import pandas as pd
import ghnn

data_path = os.path.join('..', 'Data')
num_runs = 500
store_name = 'pend_all_runs.h5.1'
nu_q = 0.95
nu_p = 0
dt = 0.01
tol, max_steps = (0.0001, 100)
seed = 0

integrator = ('Symplectic Euler', True)
integrator = ghnn.integrators.integrator_from_name(integrator[0], integrator[1])

kwargs = {'validation_share': 0.1,
          'test_share': 0.1,
          'seed': seed}
save_kwargs = {'complib': 'zlib', 'complevel': 1}

if not os.path.exists(data_path):
    os.mkdir(data_path)

ghnn.data.generate_pendulum_inputs(data_path, num_runs, store_name, nu_q=nu_q, nu_p=nu_p, seed=seed)

for run_num in range(num_runs):
    data = pd.read_hdf(os.path.join(data_path, store_name), '/run' + str(run_num)).iloc[0]
    constants = pd.read_hdf(os.path.join(data_path, store_name), '/constants')
    bodies = constants['bodies']

    calculations = []
    m = constants['mass']
    g = constants['g']
    l = constants['length']
    grad_q = ghnn.gradients.Pendulum_grad_q(m, g, l)
    grad_p = ghnn.gradients.Pendulum_grad_p(m, l)

    p = data[['p_'+body for body in bodies]].values
    q = data[['q_'+body for body in bodies]].values
    calculations.append(np.concatenate((q,p)))

    T = 1e10
    t = 0
    while t < T:
        p_old, q_old = p, q
        p, q = integrator(p_old, q_old, dt, grad_p, grad_q)

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

        if (np.sign(q) - np.sign(q_old))[0] != 0 and T == 1e10:
            T = 4 * (t + abs(q_old[0]) / (abs(q[0]) + abs(q_old[0])) * dt)

        calculations.append(np.concatenate((q,p)))
        t += dt

    columns = [qp+'_'+body for (qp, body) in product(['q', 'p'], bodies)]
    data = pd.DataFrame(calculations, columns=columns)
    data['time'] = data.index * dt
    data.to_hdf(os.path.join(data_path, store_name), key='/run' + str(run_num), format='fixed', **save_kwargs)

    constants['step_size'] = dt
    constants.to_hdf(os.path.join(data_path, store_name), key='/constants', format='fixed', **save_kwargs)

ghnn.data.combine(data_path, store_name, num_runs)

ghnn.data.create_pendulum_training_dataframe(data_path, store_name, 'h_01_training.h5.1', num_runs, 1e-1, **kwargs)

save_name = os.path.join(data_path, 'pend_training.h5.1')

for d_type in ['', 'val_', 'test_']:
    feat = pd.read_hdf(os.path.join(data_path, f'h_01_training.h5.1'), f'/{d_type}features')
    lab = pd.read_hdf(os.path.join(data_path, f'h_01_training.h5.1'), f'/{d_type}labels')
    constants = pd.read_hdf(os.path.join(data_path, f'h_01_training.h5.1'), '/constants')
    runs = feat['run'].unique()

    keep = []
    for run in runs:
        run_data = feat[feat['run'] == run]
        until = np.where(np.diff(np.sign(run_data['q_A'])))[0][0] + 1
        keep += list(run_data.iloc[:until].index)

    feat = feat.loc[keep]
    lab = lab.loc[feat.index]

    feat.to_hdf(save_name, key=f'/{d_type}features', format='fixed', **save_kwargs)
    lab.to_hdf(save_name, key=f'/{d_type}labels', format='fixed', **save_kwargs)
constants.to_hdf(save_name, key='/constants', format='fixed', **save_kwargs)

os.remove(os.path.join(data_path, 'h_01_training.h5.1'))
