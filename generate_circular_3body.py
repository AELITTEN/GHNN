import os
import ghnn

data_path = os.path.join('..', 'Data')
if not os.path.exists(data_path):
    os.mkdir(data_path)

data_path = os.path.join('..', 'Data', 'circular_3body')
num_runs = 5000
store_name = 'all_runs.h5.1'
seed = 0

kwargs = {'validation_share': 0.1,
          'test_share': 0.1,
          'seed': seed}

ghnn.data.circular_brutus(data_path, num_runs, brutus_path=os.path.join('..', 'Brutus-MPI'), T=7, dt=1e-2, seed=seed)
ghnn.data.create_training_dataframe(data_path, store_name, 'h_01_training.h5.1', num_runs, 0.1, **kwargs)
ghnn.data.create_training_dataframe(data_path, store_name, 'h_01_m5_training.h5.1', num_runs, 0.1, max_time=5, **kwargs)
ghnn.data.create_training_dataframe(data_path, store_name, 'h_05_training.h5.1', num_runs, 0.5, **kwargs)
ghnn.data.create_training_dataframe(data_path, store_name, 'h_05_m5_training.h5.1', num_runs, 0.5, max_time=5, **kwargs)

