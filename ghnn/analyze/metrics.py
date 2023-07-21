"""Functions to calculate different metrics of predictions."""
import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ghnn.nets.helpers import net_from_dir, predict_trajectories

__all__ = ['calculate_error', 'dataset_mse', 'dataset_mae']

def calculate_error(data, predictions, feature_names, interp='linear'):
    """Calulcates the error at each timestep for one trajectory using interpolation for the data.

    Args:
        data (pd.DataFrame): Data of the trajectory (including time).
        predictions (pd.DataFrame): Predcitions from an NN (including time).
        feature_names (str[]): Names of the features of the NN.
        interp (str): Type of iterpolation for times in between data points.
          Can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.

    Returns:
        pd.DataFrame: The error at each timestep for all features.
    """
    f = interp1d(data['time'].values, data[feature_names].values, axis=0, kind=interp, fill_value='extrapolate')
    error = f(predictions['time'].values) - predictions[feature_names].values
    return pd.DataFrame(error, columns=feature_names)

def dataset_mse(data_path, store_name, nn_path, max_time=None, period_q=None, t_in_T=False, interp='linear', test=None):
    """Calulcates the mean of the mean square error at each timestep for all trajectories in a dataset.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        nn_path (str): Path to where the NN is saved.
        max_time (float): Maximal time until when to calculate the error.
        period_q (float): If not None inputs are mapped to [-period_q, period_q].
        t_in_T (bool): Whether to use one period as max time.
        interp (str): Type of iterpolation for times in between data points.
          Can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.
        test (str, path-like object): HDF5 training store where the test data can be found.
          If None all data is used.

    Returns:
        pd.DataFrame: Times and mean of the MSE at each timestep.
    """
    settings = {'period_q': period_q, 'interp': interp, 'test': test, 't_in_T': t_in_T}
    if not os.path.isdir(os.path.join(nn_path, 'metrics')):
        os.makedirs(os.path.join(nn_path, 'metrics'))
    if os.path.isfile(os.path.join(nn_path, 'metrics', 'mse.txt')):
        with open(os.path.join(nn_path, 'metrics', 'mse.txt'), 'r') as file_:
            lines = file_.readlines()
        cont = [json.loads(line[12:-1]) for line in lines if line[:10]=='# Settings']
        if settings in cont:
            result = np.loadtxt(os.path.join(nn_path, 'metrics', 'mse.txt'))
            i = cont.index(settings)
            result = pd.DataFrame(result[i*2:i*2+2].T, columns=['time', 'mse'])
            if max_time != None:
                result = result[result['time'] <= max_time]
            return result

    my_net = net_from_dir(nn_path)

    data = pd.read_hdf(os.path.join(data_path, store_name), '/all_runs')
    if test:
        test_runs = pd.read_hdf(os.path.join(data_path, test), '/test_features')
        test_runs = test_runs['run'].unique()
        data = data[np.isin(data.index.get_level_values(0), test_runs)]
    inp = data.xs(0, level='timestep')

    runs = data.index.get_level_values(0).unique()
    max_time_data = [0.]*len(runs)
    for i, run in enumerate(runs):
        max_time_data[i] = data.loc[run].iloc[-1]['time']

    predictions = predict_trajectories(my_net, inp, max_time_data, period_q=period_q)

    if period_q != None:
        q = my_net.get_pos_features()
        data[q] = (data[q] + period_q) % (2 * period_q) - period_q

    if t_in_T:
        result = pd.DataFrame(np.linspace(0, 1, 101), columns=['time'])
        mean = np.zeros(101)
    else:
        ind = max_time_data.index(max(max_time_data))
        result = pd.DataFrame(predictions.loc[runs[ind]]['time'])
        mean = np.zeros(max(predictions.index.get_level_values(1))+1)

    for run in runs:
        error = calculate_error(data.loc[run], predictions.loc[run],
                                my_net.settings['feature_names'], interp=interp)

        if period_q != None:
            q = my_net.get_pos_features()
            error = error.abs()
            error[q] = period_q - (error[q] - period_q).abs()

        if t_in_T:
            f = interp1d(np.linspace(0, 1, len(error.values)), error.values, axis=0, kind=interp)
            error = pd.DataFrame(f(np.linspace(0, 1, 101)), columns=my_net.settings['feature_names'])

        error = error.values**2
        error = error.mean(axis=-1)
        mean[:len(error)] += error
    result['mse'] = mean/len(runs)

    with open(os.path.join(nn_path, 'metrics', 'mse.txt'), 'a') as file_:
        file_.write(f'# Settings: {json.dumps(settings)}\n')
        file_.write('  '.join([f'{time:.5e}' for time in result['time']])+'\n')
        file_.write('  '.join([f'{mse:.5e}' for mse in result['mse']])+'\n')

    if max_time != None:
        result = result[result['time'] <= max_time]

    return result

def dataset_mae(data_path, store_name, nn_path, max_time=None, period_q=None, t_in_T=False, interp='linear', test=None):
    """Calulcates the mean of the mean absolute error at each timestep for all trajectories in a dataset.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        nn_path (str): Path to where the NN is saved.
        max_time (float): Maximal time until when to calculate the error.
        period_q (float): If not None inputs are mapped to [-period_q, period_q].
        t_in_T (bool): Whether to use one period as max time.
        interp (str): Type of iterpolation for times in between data points.
          Can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.
        test (str, path-like object): HDF5 training store where the test data can be found.
          If None all data is used.

    Returns:
        pd.DataFrame: Times and mean of the MAE at each timestep.
    """
    settings = {'period_q': period_q, 'interp': interp, 'test': test, 't_in_T': t_in_T}
    if not os.path.isdir(os.path.join(nn_path, 'metrics')):
        os.makedirs(os.path.join(nn_path, 'metrics'))
    if os.path.isfile(os.path.join(nn_path, 'metrics', 'mae.txt')):
        with open(os.path.join(nn_path, 'metrics', 'mae.txt'), 'r') as file_:
            lines = file_.readlines()
        cont = [json.loads(line[12:-1]) for line in lines if line[:10]=='# Settings']
        if settings in cont:
            result = np.loadtxt(os.path.join(nn_path, 'metrics', 'mae.txt'))
            i = cont.index(settings)
            result = pd.DataFrame(result[i*2:i*2+2].T, columns=['time', 'mae'])
            if max_time != None:
                result = result[result['time'] <= max_time]
            return result

    my_net = net_from_dir(nn_path)

    data = pd.read_hdf(os.path.join(data_path, store_name), '/all_runs')
    if test:
        test_runs = pd.read_hdf(os.path.join(data_path, test), '/test_features')
        test_runs = test_runs['run'].unique()
        data = data[np.isin(data.index.get_level_values(0), test_runs)]
    inp = data.xs(0, level='timestep')

    runs = data.index.get_level_values(0).unique()
    max_time_data = [0.]*len(runs)
    for i, run in enumerate(runs):
        max_time_data[i] = data.loc[run].iloc[-1]['time']

    predictions = predict_trajectories(my_net, inp, max_time_data, period_q=period_q)

    if period_q != None:
        q = my_net.get_pos_features()
        data[q] = (data[q] + period_q) % (2 * period_q) - period_q

    if t_in_T:
        result = pd.DataFrame(np.linspace(0, 1, 101), columns=['time'])
        mean = np.zeros(101)
    else:
        ind = max_time_data.index(max(max_time_data))
        result = pd.DataFrame(predictions.loc[runs[ind]]['time'])
        mean = np.zeros(max(predictions.index.get_level_values(1))+1)

    for run in runs:
        error = calculate_error(data.loc[run], predictions.loc[run],
                                my_net.settings['feature_names'], interp=interp)

        if period_q != None:
            q = my_net.get_pos_features()
            error = error.abs()
            error[q] = period_q - (error[q] - period_q).abs()

        if t_in_T:
            f = interp1d(np.linspace(0, 1, len(error.values)), error.values, axis=0, kind=interp)
            error = pd.DataFrame(f(np.linspace(0, 1, 101)), columns=my_net.settings['feature_names'])

        error = np.abs(error.values)
        error = error.mean(axis=-1)
        mean[:len(error)] += error
    result['mae'] = mean/len(runs)

    with open(os.path.join(nn_path, 'metrics', 'mae.txt'), 'a') as file_:
        file_.write(f'# Settings: {json.dumps(settings)}\n')
        file_.write('  '.join([f'{time:.5e}' for time in result['time']])+'\n')
        file_.write('  '.join([f'{mae:.5e}' for mae in result['mae']])+'\n')

    if max_time != None:
        result = result[result['time'] <= max_time]

    return result
