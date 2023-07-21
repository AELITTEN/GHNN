import os
import ghnn

data_path = os.path.join('..', 'Data')
if not os.path.exists(data_path):
    os.mkdir(data_path)

data_path = os.path.join('..', 'Data', 'double_pendulum')
num_runs = 2000
store_name = 'all_runs.h5.1'
nu_q = 0.5
nu_p = 0
T = 10
dt = 0.01
converge = (0.001, 100)
seed = 0

kwargs = {'validation_share': 0.1,
          'test_share': 0.1,
          'seed': seed}

ghnn.data.double_pendulum(data_path, num_runs, store_name=store_name, nu_q=nu_q, nu_p=nu_p, T=T, dt=dt, converge=converge, seed=seed)
ghnn.data.create_pendulum_training_dataframe(data_path, store_name, 'h_01_training.h5.1', num_runs, 1e-1, **kwargs)
ghnn.data.create_pendulum_training_dataframe(data_path, store_name, 'h_05_training.h5.1', num_runs, 5e-1, **kwargs)
ghnn.data.create_pendulum_training_dataframe(data_path, store_name, 'h_01_m5_training.h5.1', num_runs, 1e-1, max_time=5, **kwargs)
ghnn.data.create_pendulum_training_dataframe(data_path, store_name, 'h_05_m5_training.h5.1', num_runs, 5e-1, max_time=5, **kwargs)
