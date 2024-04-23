import os
import ghnn

data_path = os.path.join('..', 'Data')
num_runs = 5000
store_name = 'circ_3body_all_runs.h5.1'
T = 7
dt = 0.01
converge = (0.001, 100)
seed = 0

kwargs = {'validation_share': 0.1,
          'test_share': 0.1,
          'seed': seed}

#ghnn.data.circular_brutus(data_path, num_runs, brutus_path=os.path.join('..', 'Brutus-MPI'), store_name=store_name, T=T, dt=dt, seed=seed)
ghnn.data.circular(data_path, num_runs, store_name=store_name, T=T, dt=dt, converge=converge, seed=seed)
ghnn.data.create_training_dataframe(data_path, store_name, 'circ_3body_training.h5.1', num_runs, 0.5, max_time=5, **kwargs)

