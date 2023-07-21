"""A few functions to map from brutus or integrator data to different pandas.DataFrames.

The functions in here are typically used in the folowing order
after data has been created using the functions in ghnn.data.generate_data:

Example:
    >>> data_path = '<path-to-data>'
    >>> store_name = 'test.h5.1'
    >>> num_runs = 100
    >>> step_size = 0.1
    >>> max_time = 5
    >>> training_store_name = 'test_h_01_training.h5.1'
    >>> feature_names = ['q_A_x', 'q_A_y', 'q_B_x', 'q_B_y', 'q_C_x', 'q_C_y', 'p_A_x', 'p_A_y', 'p_B_x', 'p_B_y', 'p_C_x', 'p_C_y']
    >>> label_names = ['q_A_x', 'q_A_y', 'q_B_x', 'q_B_y', 'q_C_x', 'q_C_y', 'p_A_x', 'p_A_y', 'p_B_x', 'p_B_y', 'p_C_x', 'p_C_y']
    >>> validation_share = 0.1
    >>> test_share = 0.1
    >>> extract_from_brutus(data_path, store_name, num_runs)
    >>> combine(data_path, store_name, num_runs)
    >>> create_training_dataframe(data_path, store_name, training_store_name, num_runs, step_size, feature_names, label_names, validation_share, test_share, max_time)
"""
import string
import os
import subprocess
import time
import csv
import json
from itertools import product
#import pickle
#pickle.HIGHEST_PROTOCOL = 4
import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

__all__ = ['extract_from_brutus', 'combine', 'create_training_dataframe', 'create_pendulum_training_dataframe']

def extract_from_brutus(data_path, store_name, num_runs, scale='SI', save_dims=['x', 'y', 'z']):
    """Extracts runs from brutus data and saves them to a HDF5 store.

    Assumes the brutus data is in a sub directory caled 'brutus_runs' of the data_path.
    All runs are assumed to be saved in individual files called 'run{i}.diag'.
    The nuber of bodies N is automatically extracted from 'run0.log' and all bodies are named: A, B, C, ...
    All runs are saved as independent DataFrames to the store.
    'bodies', 'dimensions', 'masses' and 'step_size' are saved as a constants DataFrame in the HDF5 store.

    Args:
        data_path (str, path-like object): Path to where the brutus data lies
          in a subdir called 'brutus_runs' and where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        num_runs (int): Number of runs that are supposed to be extracted.
        scale (str): To which units should the data be scaled. ('SI', 'viralnbody' or 'keplernbody').
        save_dims (str[]): Dimensions that are supposed to be saved.
    """
    with open(os.path.join(data_path, 'brutus_runs', 'run0.log'), newline='') as file_:
        logs = {}
        start_logs = False
        for line in file_:
            if line[0] == 'N':
                start_logs = True
            if start_logs:
                logs[line[:line.index(' ')]] = line[line.index('=')+2:-1]
    with open(os.path.join(data_path, 'brutus_runs', 'scales0.json'), 'r') as file_:
        scales = json.load(file_)

    if scale == 'viralnbody':
        scales['time'] = 1
    elif scale == 'keplernbody':
        with open(os.path.join(data_path, 'brutus_runs', 'run0.dat'), newline='') as file_:
            lines = file_.read().splitlines()
        m1 = float(lines[1].split(' ')[0])
        m2 = float(lines[2].split(' ')[0])
        scales['time'] = math.sqrt((m1+m2) * pow(scales['position']/scales['semi_major_axis'], 3))

    step_size = float(logs['dt']) * scales['time']
    dimensions = ['x', 'y', 'z']
    bodies = list(string.ascii_uppercase)[:int(logs['N'])]
    columns = [qp+'_'+body+'_'+dim for (body,qp,dim) in product(bodies, ['q', 'p'], dimensions)]
    save_columns = [qp+'_'+body+'_'+dim for (qp,body,dim) in product(['q', 'p'], bodies, save_dims)]

    kwargs = {'complib': 'zlib', 'complevel': 1}
    masses = [[0] * len(bodies)] * num_runs
    for i in range(num_runs):
        with open(os.path.join(data_path, 'brutus_runs', f'run{i}.diag'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            timestep = []
            data = []
            for row in reader:
                if len(row) == 8:
                    timestep += row[:-1]
                elif len(row) == 0:
                    data.append(timestep)
                    timestep = []

        with open(os.path.join(data_path, 'brutus_runs', f'scales{i}.json'), 'r') as file_:
            scales = json.load(file_)
        if scale == 'viralnbody':
            scales['mass'] = 1
            scales['velocity'] = 1
            scales['position'] = 1
        elif scale == 'keplernbody':
            scales['position'] /= (1/(m1+m2)) ** (1/3) * scales['semi_major_axis']
            scales['velocity'] /= math.sqrt(6.67428e-11*scales['mass']/((1/(m1+m2)) ** (1/3) * scales['semi_major_axis']))
            scales['mass'] = 1

        data = np.array(data[:-1])
        data = data.astype(float)
        mask = [True] * data.shape[1]
        for j in range(len(bodies)):
            mask[j*(len(dimensions)*2+1)] = False
            masses[i][j] = data[0][j*(len(dimensions)*2+1)] * scales['mass']

        data = pd.DataFrame(data[:,mask], columns=columns)
        data = data[save_columns]
        for j, body in enumerate(bodies):
            data[[col for col in save_columns if col[0:3] == 'p_'+body]] *= scales['velocity'] * masses[i][j]
        data[[col for col in save_columns if col[0] == 'q']] *= scales['position']
        data['time'] = data.index * step_size
        data.to_hdf(os.path.join(data_path, store_name), '/run' + str(i), format='fixed', **kwargs)

    constants = pd.Series([bodies, save_dims, masses], index=['bodies', 'dimensions', 'masses'])
    constants['step_size'] = step_size
    constants['scale'] = scale
    git_dir = os.path.dirname(__file__)[:-9] + '.git'
    version = subprocess.check_output(['git',
                                       '--git-dir=' + git_dir,
                                       'rev-parse', 'HEAD']).decode('ascii').strip()
    constants['code_version'] = version
    constants.to_hdf(os.path.join(data_path, store_name), '/constants', format='fixed', **kwargs)

def combine(data_path, store_name, num_runs):
    """Combines DataFrames from several runs into one large DataFrame.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        num_runs (int): Number of runs that are supposed to be combined.
    """
    data_path = os.path.join(data_path, store_name)
    runs = {}
    for i in range(num_runs):
        if i % 100 == 0:
            print('Now at ' + str(i))
        runs[i] = pd.read_hdf(data_path, '/run' + str(i)).rename_axis("timestep")
    all_runs = pd.concat(runs, names=['run'])

    kwargs = {'complib': 'zlib', 'complevel': 1}
    all_runs.to_hdf(data_path, '/all_runs', format='fixed', **kwargs)

def create_training_dataframe(data_path, store_name, training_store_name, num_runs, step_size, feature_names=None, label_names=None, validation_share=0.1, test_share=0.1, max_time=None, shift_t0=False, interp='linear', seed=None):
    """Creates from a combined DataFrame two (features, labels) DataFrames with constant time steps suitable for training.

    For the creation of training data for timestepped NNs. Like SympNets.
    Time is never included in the feature names.
    Data between data points are calulcated using scipy.interpolate.interp1d.
    Copies constants from the original store but overrites 'step_size'.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store with all runs.
        training_store_name (str, path-like object): Name of the HDF5 store for the training data.
        num_runs (int): Number of runs that are supposed to be combined.
        step_size (float): Size of the time steps.
        feature_names (str[]): Column names that are supposed to be in the features DataFrame.
        label_names (str[]): Column names that are supposed to be in the labels DataFrame.
        validation_share (float): Fraction of the data that should be used for validation. Between 0 and 1-test_share.
        test_share (float): Fraction of the data that should be used for testing. Between 0 and 1-validation_share.
        max_time (float): End time for the runs.
        shift_t0 (bool): Whether to shift t0 to extract all data from one trajectory.
        interp (str): Type of iterpolation for times in between data points.
          Can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.
        seed (int): Random seed for split in train/validation/test data.
    """
    load_name = os.path.join(data_path, store_name)
    data = pd.read_hdf(load_name, '/all_runs')
    constants = pd.read_hdf(load_name, '/constants')

    if not feature_names:
        feature_names = list(data.columns)
    if 'time' in feature_names:
        feature_names.remove('time')

    if not label_names:
        label_names = list(data.columns)
        if 'time' in label_names:
            label_names.remove('time')

    train_features = []
    train_labels = []

    t0 = time.time()
    start_time = t0
    total_time = 0
    for i in range(num_runs):
        if i % 100 == 0 and i != 0:
            print('Now at ' + str(i))
            time_spend = time.time() - t0
            t0 = time.time()
            total_time += time_spend
            print('The last 100 runs took', time_spend/60, 'minutes.')
            print('Probably', total_time/i*(num_runs-i)/60, 'minutes left.')
        run = data.loc[i]
        f = interp1d(run['time'].values, run[label_names].values, axis=0, kind=interp, fill_value='extrapolate')

        if shift_t0:
            x0s = np.arange(0, step_size, constants['step_size'])
        else:
            x0s=[0]

        for x0 in x0s:
            if max_time:
                x = np.arange(x0, min(max_time, run['time'].iloc[-1]) + constants['step_size']/2, step_size)
            else:
                x = np.arange(x0, run['time'].iloc[-1] + constants['step_size']/2, step_size)
            if x[-1] > run['time'].iloc[-1]:
                x = np.delete(x, -1)
            run_index = np.array([[i]] * (len(x)-1))
            dist1 = step_size%constants['step_size']
            dist2 = min(dist1, abs(dist1-constants['step_size']))
            if (dist2 * len(x)) > (0.1*constants['step_size']):
                print('Warning heavily interpolating the data!!!')
            if len(x) > 1:
                train_features.append(np.concatenate((run_index, f(x[:-1])), axis=1))
                train_labels.append(f(x[1:]))

    train_features = pd.DataFrame(np.concatenate(train_features, axis = 0),
                                  columns=['run']+feature_names).astype({'run':int})
    train_labels = pd.DataFrame(np.concatenate(train_labels, axis = 0), columns=label_names)

    if seed != None:
        np.random.seed(seed)
    perm = np.arange(num_runs)
    np.random.shuffle(perm)
    validation_runs = np.isin(train_features['run'], perm[:int(validation_share*num_runs)])
    test_runs = np.isin(train_features['run'], perm[int(validation_share*num_runs):
                                                    int((validation_share+test_share)*num_runs)])
    train_runs = np.isin(train_features['run'], perm[int((validation_share+test_share)*num_runs):])

    validation_features = train_features[validation_runs]
    validation_labels = train_labels[validation_runs]
    test_features = train_features[test_runs]
    test_labels = train_labels[test_runs]
    train_features = train_features[train_runs]
    train_labels = train_labels[train_runs]

    print('Total time for' , num_runs, 'runs is', (time.time() - start_time) / 60, 'minutes.')

    constants['step_size'] = float(step_size)
    git_dir = os.path.dirname(__file__)[:-9] + '.git'
    version = subprocess.check_output(['git',
                                       '--git-dir=' + git_dir,
                                       'rev-parse', 'HEAD']).decode('ascii').strip()
    constants['code_version'] = version

    kwargs = {'complib': 'zlib', 'complevel': 1}
    save_name = os.path.join(data_path, training_store_name)
    constants.to_hdf(save_name, '/constants', format='fixed', **kwargs)
    validation_features.to_hdf(save_name, '/val_features', format='fixed', **kwargs)
    validation_labels.to_hdf(save_name, '/val_labels', format='fixed', **kwargs)
    test_features.to_hdf(save_name, '/test_features', format='fixed', **kwargs)
    test_labels.to_hdf(save_name, '/test_labels', format='fixed', **kwargs)
    train_features.to_hdf(save_name, '/features', format='fixed', **kwargs)
    train_labels.to_hdf(save_name, '/labels', format='fixed', **kwargs)

def create_pendulum_training_dataframe(data_path, store_name, training_store_name, num_runs, step_size, feature_names=None, label_names=None, validation_share=0.1, test_share=0.1, max_time=None, shift_t0=False, interp='linear', seed=None):
    """Creates from a combined DataFrame two (features, labels) DataFrames with constant time steps suitable for training.

    For the creation of training data for timestepped NNs. Like SympNets.
    Time is never included in the feature names.
    Data between data points are calulcated using scipy.interpolate.interp1d.
    Respects periodicity in q for pendulum data.
    Copies constants from the original store but overrites 'step_size'.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store with all runs.
        training_store_name (str, path-like object): Name of the HDF5 store for the training data.
        num_runs (int): Number of runs that are supposed to be combined.
        step_size (float): Size of the time steps.
        feature_names (str[]): Column names that are supposed to be in the features DataFrame.
        label_names (str[]): Column names that are supposed to be in the labels DataFrame.
        validation_share (float): Fraction of the data that should be used for validation. Between 0 and 1-test_share.
        test_share (float): Fraction of the data that should be used for testing. Between 0 and 1-validation_share.
        max_time (float): End time for the runs.
        shift_t0 (bool): Whether to shift t0 to extract all data from one trajectory.
        interp (str): Type of iterpolation for times in between data points.
          Can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.
        seed (int): Random seed for split in train/validation/test data.
    """
    load_name = os.path.join(data_path, store_name)
    data = pd.read_hdf(load_name, '/all_runs')
    constants = pd.read_hdf(load_name, '/constants')

    if not feature_names:
        feature_names = list(data.columns)
    if 'time' in feature_names:
        feature_names.remove('time')

    if not label_names:
        label_names = list(data.columns)
        if 'time' in label_names:
            label_names.remove('time')

    train_features = []
    train_labels = []

    t0 = time.time()
    start_time = t0
    total_time = 0
    for i in range(num_runs):
        if i % 100 == 0 and i != 0:
            print('Now at ' + str(i))
            time_spend = time.time() - t0
            t0 = time.time()
            total_time += time_spend
            print('The last 100 runs took', time_spend/60, 'minutes.')
            print('Probably', total_time/i*(num_runs-i)/60, 'minutes left.')
        run = data.loc[i]
        f = interp1d(run['time'].values, run[label_names].values, axis=0, kind=interp, fill_value='extrapolate')

        if shift_t0:
            x0s = np.arange(0, step_size, constants['step_size'])
        else:
            x0s=[0]

        for x0 in x0s:
            if max_time:
                x = np.arange(x0, min(max_time, run['time'].iloc[-1]) + constants['step_size']/2, step_size)
            else:
                x = np.arange(x0, run['time'].iloc[-1] + constants['step_size']/2, step_size)
            run_index = np.array([[i]] * (len(x)-1))
            dist1 = step_size%constants['step_size']
            dist2 = min(dist1, abs(dist1-constants['step_size']))
            if (dist2 * len(x)) > (0.1*constants['step_size']):
                print('Warning heavily interpolating the data!!!')
            if len(x) > 1:
                train_features.append(np.concatenate((run_index, f(x[:-1])), axis=1))
                train_labels.append(f(x[1:]))

    train_features = pd.DataFrame(np.concatenate(train_features, axis = 0),
                                  columns=['run']+feature_names).astype({'run':int})
    train_labels = pd.DataFrame(np.concatenate(train_labels, axis = 0), columns=label_names)

    for body in constants['bodies']:
        g_pi = train_features[train_features['q_'+body] > math.pi].index
        mult = ((train_features.loc[g_pi]['q_'+body] + math.pi) / (2 * math.pi)).astype(int)
        train_features.loc[g_pi,'q_'+body] -= mult * 2 * math.pi
        train_labels.loc[g_pi,'q_'+body] -= mult * 2 * math.pi

        s_pi = train_features[train_features['q_'+body] < -math.pi].index
        mult = ((train_features.iloc[s_pi]['q_'+body] - math.pi) / (2 * math.pi)).astype(int)
        train_features.loc[s_pi,'q_'+body] -= mult * 2 * math.pi
        train_labels.loc[s_pi,'q_'+body] -= mult * 2 * math.pi

    if seed != None:
        np.random.seed(seed)
    perm = np.arange(num_runs)
    np.random.shuffle(perm)
    validation_runs = np.isin(train_features['run'], perm[:int(validation_share*num_runs)])
    test_runs = np.isin(train_features['run'], perm[int(validation_share*num_runs):
                                                    int((validation_share+test_share)*num_runs)])
    train_runs = np.isin(train_features['run'], perm[int((validation_share+test_share)*num_runs):])

    validation_features = train_features[validation_runs]
    validation_labels = train_labels[validation_runs]
    test_features = train_features[test_runs]
    test_labels = train_labels[test_runs]
    train_features = train_features[train_runs]
    train_labels = train_labels[train_runs]

    print('Total time for' , num_runs, 'runs is', (time.time() - start_time) / 60, 'minutes.')

    constants['step_size'] = float(step_size)
    git_dir = os.path.dirname(__file__)[:-9] + '.git'
    version = subprocess.check_output(['git',
                                       '--git-dir=' + git_dir,
                                       'rev-parse', 'HEAD']).decode('ascii').strip()
    constants['code_version'] = version

    kwargs = {'complib': 'zlib', 'complevel': 1}
    save_name = os.path.join(data_path, training_store_name)
    constants.to_hdf(save_name, '/constants', format='fixed', **kwargs)
    validation_features.to_hdf(save_name, '/val_features', format='fixed', **kwargs)
    validation_labels.to_hdf(save_name, '/val_labels', format='fixed', **kwargs)
    test_features.to_hdf(save_name, '/test_features', format='fixed', **kwargs)
    test_labels.to_hdf(save_name, '/test_labels', format='fixed', **kwargs)
    train_features.to_hdf(save_name, '/features', format='fixed', **kwargs)
    train_labels.to_hdf(save_name, '/labels', format='fixed', **kwargs)
