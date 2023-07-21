"""Plotting predictions of trained NNs.

Example:
    >>> data_path = '<path-to-data>'
    >>> store_name = 'all_runs.h5.1'
    >>> num_runs = 5000
    >>> nn_path = '<path-to-nn>'
    >>> predict_run_rand(data_path, store_name, num_runs, nn_path)
"""
import os
import math
import random
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from ghnn.plotting.helpers import plot_2D, plot_3D, save_show
from ghnn.analyze.nbody_energy import nbody_all_energy
from ghnn.analyze.pendulum_energy import pendulum_all_energy, doublependulum_all_energy
from ghnn.analyze.momentum import angular_momentum
from ghnn.analyze.metrics import calculate_error
from ghnn.nets.helpers import net_from_dir, predict_trajectories
from ghnn.constants import G

__all__ = ['predict_run', 'predict_run_rand', 'predict_pendulum', 'predict_pendulum_rand']

def predict_run(data_path, store_name, run_num, nn_paths, save_name=None, plot_data=True, max_time=None, phase_space=False, energy=False, momentum=False, mse=False, d3=False, dims=['x','y'], device=None):
    """Plots the predictions of one or several NNs of one N-body trajectory using the initial state from data.

    Data and NNs should all share the same scaling!
    Plots the predicted trajectories in 3D or 2D. With or without data.
    Also plots the energy and/or the angular momentum if desired.
    The plot can be saved or shown to the user.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        run_num (int): Index of the trajectory.
        nn_paths (str, str[]): Path(s) to where the NN(s) are saved.
        save_name (str, path-like,  binary file-like): Path where to save the plot.
          If None then the plot is shown and not saved.
        plot_data (bool): Whether to also plot the data.
        max_time (float): Maximal time until when to plot the trajectory.
        phase_space (bool): Whether to plot postions and momenta additionally over time.
        energy (bool): Whether to also plot the energy.
        momentum (bool): Whether to also plot the angular momentum.
        mse (bool): Whether to also plot the MSE.
        d3 (bool): Switch for a 3D or 2D plot.
        dims (str[]): Which two dimensions should be plotted, in case d3==False.
        device (str): The device to do the computations. 'cpu' or 'gpu'.
          If None settings['device'] is used.
    """
    if not isinstance(nn_paths, list):
        nn_paths = [nn_paths]
    num_nets = len(nn_paths)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.suptitle(f'Run number {run_num}:')
    width = 1
    if phase_space:
        width += 2
    if energy:
        width += 1
    if momentum:
        width += 1
    if mse:
        width += 1

    if width == 1:
        height = math.floor(math.sqrt(num_nets))
        width = math.ceil(num_nets/height)
    else:
        height = num_nets

    constants = pd.read_hdf(os.path.join(data_path, store_name), '/constants')
    data = pd.read_hdf(os.path.join(data_path, store_name), '/run' + str(run_num))
    max_time_data = data['time'].iloc[-1]
    if max_time:
        data = data[data['time']<=max_time]
        max_time_data = data['time'].iloc[-1]
    else:
        max_time = max_time_data
    positions = data[[col for col in data if col[0]=='q']]
    momenta = data[[col for col in data if col[0]=='p']]

    subplt = 0
    for i, nn_path in enumerate(nn_paths):
        subplt += 1
        my_net = net_from_dir(nn_path, device=device)

        start_time = time.time()
        predictions = predict_trajectories(my_net, data.loc[0], max_time)
        total_time = time.time() - start_time
        print('Time spent to predict one path: ', total_time)

        pred_times = np.linspace(0, max_time, predictions.shape[0])

        if d3:
            ax = fig.add_subplot(height, width, subplt, projection='3d')
            plot_3D(ax, constants['bodies'], predictions, symbol='--')
            if plot_data:
                plot_3D(ax, constants['bodies'], data, init=False)
        else:
            ax = fig.add_subplot(height, width, subplt)
            plot_2D(ax, constants['bodies'], predictions, dims=dims, symbol='--')
            if plot_data:
                plot_2D(ax, constants['bodies'], data, dims=dims, init=False)

        path = os.path.normpath(nn_path).split(os.path.sep)
        if path[-1][:2] == 'nn':
            name = path[-2]
        else:
            name = path[-1]
        ax.set_ylabel(name)

        if i == 0:
            ax.set_title('Trajectories:')

        if phase_space:
            colors = ['r', 'b', 'g', 'm', 'y', 'c', 'k']
            subplt += 1
            ax = fig.add_subplot(height, width, subplt)
            if i == 0:
                ax.set_title('Positions:')
            for j, q in enumerate(positions):
                ax.plot(data['time'], positions[q], colors[(j%len(colors))]+'-', label=q)
                ax.plot(pred_times, predictions[q], colors[(j%len(colors))]+'--')
            ax.legend()

            subplt += 1
            ax = fig.add_subplot(height, width, subplt)
            if i == 0:
                ax.set_title('Momenta:')
            for j, p in enumerate(momenta):
                ax.plot(data['time'], momenta[p], colors[(j%len(colors))]+'-', label=p)
                ax.plot(pred_times, predictions[p], colors[(j%len(colors))]+'--')
            ax.legend()

        if energy:
            subplt += 1
            ax = fig.add_subplot(height, width, subplt)
            if i == 0:
                ax.set_title('Energy:')
                if constants['scale'] == 'SI':
                    g = G
                else:
                    g = 1
            if plot_data:
                ke, pe = nbody_all_energy(data, constants['bodies'], constants['dimensions'], constants['masses'][0], g=g)
                ax.plot(np.linspace(0, max_time_data, len(ke)), ke, 'b-')
                ax.plot(np.linspace(0, max_time_data, len(pe)), pe, 'g-')
                ax.plot(np.linspace(0, max_time_data, len(ke)), ke+pe, 'r-')
            pred_ke, pred_pe = nbody_all_energy(predictions, constants['bodies'], constants['dimensions'], constants['masses'][0], g=g)
            ax.plot(np.linspace(0, max_time, len(pred_ke)), pred_ke, 'b--')
            ax.plot(np.linspace(0, max_time, len(pred_pe)), pred_pe, 'g--')
            ax.plot(np.linspace(0, max_time, len(pred_pe)), pred_pe+pred_ke, 'r--')

        if momentum:
            subplt += 1
            ax = fig.add_subplot(height, width, subplt)
            if i == 0:
                ax.set_title('Angular Momentum:')
            if plot_data:
                am = angular_momentum(data, constants['bodies'], constants['dimensions'], constants['masses'][0])
                ax.plot(np.linspace(0, max_time_data, len(am)), am, 'r-')
            pred_am = angular_momentum(predictions, constants['bodies'], constants['dimensions'], constants['masses'][0])
            ax.plot(np.linspace(0, max_time, len(pred_am)), pred_am, 'r--')

        if mse:
            subplt += 1
            ax = fig.add_subplot(height, width, subplt)
            if i == 0:
                ax.set_title('Mean square error:')
            error = calculate_error(data, predictions[predictions['time']<=max_time_data],
                                    my_net.settings['feature_names'])
            error = error.values**2
            ax.plot(np.linspace(0, max_time_data, len(error)), error.mean(axis=-1), 'r')

    save_show(fig, save_name)

def predict_run_rand(data_path, store_name, num_runs, nn_paths, **kwargs):
    """Plots the predictions of one or several NNs of one N-body trajectory using a random initial state from data in infinit loop.

    Data and NNs should all share the same scaling!

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        num_runs (int): Number of runs from which to draw.
        nn_paths (str, str[]): Paths to where the NNs are saved.
    """
    while True:
        run_num = random.randint(0, num_runs-1)
        predict_run(data_path, store_name, run_num, nn_paths, **kwargs)

def predict_pendulum(data_path, store_name, run_num, nn_paths, save_name=None, plot_data=True, max_time=None, phase_space=False, energy=False, mse=False, period_q=None, device=None):
    """Plots the predictions of one or several NNs of one (single/double) pendulum trajectory using the initial state from data.

    Data and NNs should all share the same scaling!
    Plots the predicted trajectories. With or without data.
    Also plots the energy if desired.
    The plot can be saved or shown to the user.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        run_num (int): Index of the trajectory.
        nn_paths (str, str[]): Path(s) to where the NN(s) are saved.
        save_name (str, path-like,  binary file-like): Path where to save the plot.
          If None then the plot is shown and not saved.
        plot_data (bool): Whether to also plot the data.
        max_time (float): Maximal time until when to plot the trajectory.
        phase_space (bool): Whether to plot postions and momenta additionally over time.
        energy (bool): Whether to also plot the energy.
        mse (bool): Whether to also plot the MSE.
        period_q (float): If not None inputs are mapped to [-period_q, period_q].
        device (str): The device to do the computations. 'cpu' or 'gpu'.
          If None settings['device'] is used.
    """
    if not isinstance(nn_paths, list):
        nn_paths = [nn_paths]
    num_nets = len(nn_paths)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.suptitle(f'Run number {run_num}:')
    width = 1
    if phase_space:
        width += 2
    if energy:
        width += 1
    if mse:
        width += 1

    if width == 1:
        height = math.floor(math.sqrt(num_nets))
        width = math.ceil(num_nets/height)
    else:
        height = num_nets

    constants = pd.read_hdf(os.path.join(data_path, store_name), '/constants')
    bodies = constants['bodies']
    data = pd.read_hdf(os.path.join(data_path, store_name), '/run' + str(run_num))
    max_time_data = data['time'].iloc[-1]
    if max_time:
        data = data[data['time']<=max_time]
        max_time_data = data['time'].iloc[-1]
    else:
        max_time = max_time_data

    q = [col for col in data if col[0]=='q']
    if period_q != None:
        data[q] = (data[q] + period_q) % (2 * period_q) - period_q
    positions = data[q]
    momenta = data[[col for col in data if col[0]=='p']]

    subplt = 0
    for i, nn_path in enumerate(nn_paths):
        subplt += 1
        my_net = net_from_dir(nn_path, device=device)

        start_time = time.time()
        predictions = predict_trajectories(my_net, data.loc[0], max_time, period_q=period_q)
        total_time = time.time() - start_time
        print('Time spent to predict one path: ', total_time)

        pred_times = np.linspace(0, max_time, predictions.shape[0])

        ax = fig.add_subplot(height, width, subplt)

        if len(bodies) == 1:
            data['q_'+bodies[0]+'_x'] = constants['length'] * np.sin(data['q_'+bodies[0]])
            data['q_'+bodies[0]+'_y'] = - constants['length'] * np.cos(data['q_'+bodies[0]])
            predictions['q_'+bodies[0]+'_x'] = constants['length'] * np.sin(predictions['q_'+bodies[0]])
            predictions['q_'+bodies[0]+'_y'] = - constants['length'] * np.cos(predictions['q_'+bodies[0]])
            ax.plot([0, data['q_'+bodies[0]+'_x'].values[0]], [0, data['q_'+bodies[0]+'_y'].values[0]], 'k-')
        elif len(bodies) == 2:
            data['q_'+bodies[0]+'_x'] = constants['lengths'][0] * np.sin(data['q_'+bodies[0]])
            data['q_'+bodies[0]+'_y'] = - constants['lengths'][0] * np.cos(data['q_'+bodies[0]])
            data['q_'+bodies[1]+'_x'] = constants['lengths'][1] * np.sin(data['q_'+bodies[1]]) + data['q_'+bodies[0]+'_x']
            data['q_'+bodies[1]+'_y'] = - constants['lengths'][1] * np.cos(data['q_'+bodies[1]]) + data['q_'+bodies[0]+'_y']
            predictions['q_'+bodies[0]+'_x'] = constants['lengths'][0] * np.sin(predictions['q_'+bodies[0]])
            predictions['q_'+bodies[0]+'_y'] = - constants['lengths'][0] * np.cos(predictions['q_'+bodies[0]])
            predictions['q_'+bodies[1]+'_x'] = constants['lengths'][1] * np.sin(predictions['q_'+bodies[1]]) + predictions['q_'+bodies[0]+'_x']
            predictions['q_'+bodies[1]+'_y'] = - constants['lengths'][1] * np.cos(predictions['q_'+bodies[1]]) + predictions['q_'+bodies[0]+'_y']
            ax.plot([0, data['q_'+bodies[0]+'_x'].values[0]], [0, data['q_'+bodies[0]+'_y'].values[0]], 'k-')
            ax.plot([data['q_'+bodies[0]+'_x'].values[0], data['q_'+bodies[1]+'_x'].values[0]],
                    [data['q_'+bodies[0]+'_y'].values[0], data['q_'+bodies[1]+'_y'].values[0]], 'k-')
        else:
            raise ValueError

        plot_2D(ax, bodies, predictions, symbol='--')
        if plot_data:
            plot_2D(ax, bodies, data, init=False)

        path = os.path.normpath(nn_path).split(os.path.sep)
        if path[-1][:2] == 'nn':
            name = path[-2]
        else:
            name = path[-1]
        ax.set_ylabel(name)

        if i == 0:
            ax.set_title('Trajectories:')

        if phase_space:
            colors = ['r', 'b', 'g', 'm', 'y', 'c', 'k']
            subplt += 1
            ax = fig.add_subplot(height, width, subplt)
            if i == 0:
                ax.set_title('Positions:')
            for j, q in enumerate(positions):
                k_old = 0
                for k, _ in enumerate(data.loc[1:,'time'], start=1):
                    if abs(positions.loc[k-1, q] - positions.loc[k, q]) > 4:
                        ax.plot(data.loc[k_old:k-1, 'time'], positions.loc[k_old:k-1, q], colors[(j%len(colors))]+'-')
                        k_old = k
                ax.plot(data.loc[k_old:, 'time'], positions.loc[k_old:, q], colors[(j%len(colors))]+'-', label=q)
                k_old = 0
                for k, _ in enumerate(pred_times[1:], start=1):
                    if abs(predictions.loc[k-1, q] - predictions.loc[k, q]) > 4:
                        ax.plot(pred_times[k_old:k], predictions.loc[k_old:k-1, q], colors[(j%len(colors))]+'--')
                        k_old = k
                ax.plot(pred_times[k_old:], predictions.loc[k_old:, q], colors[(j%len(colors))]+'--')
            ax.legend()

            subplt += 1
            ax = fig.add_subplot(height, width, subplt)
            if i == 0:
                ax.set_title('Momenta:')
            for j, p in enumerate(momenta):
                ax.plot(data['time'], momenta[p], colors[(j%len(colors))]+'-', label=p)
                ax.plot(pred_times, predictions[p], colors[(j%len(colors))]+'--')
            ax.legend()

        if energy:
            subplt += 1
            ax = fig.add_subplot(height, width, subplt)
            if i == 0:
                ax.set_title('Energy:')
                g = constants['g']
            if len(bodies) == 1:
                if plot_data:
                    ke, pe = pendulum_all_energy(data, constants['mass'], constants['length'], g=g)
                    ax.plot(np.linspace(0, max_time_data, len(ke)), ke, 'b-')
                    ax.plot(np.linspace(0, max_time_data, len(pe)), pe, 'g-')
                    ax.plot(np.linspace(0, max_time_data, len(ke)), ke+pe, 'r-')
                pred_ke, pred_pe = pendulum_all_energy(predictions, constants['mass'], constants['length'], g=g)
                ax.plot(np.linspace(0, max_time, len(pred_ke)), pred_ke, 'b--')
                ax.plot(np.linspace(0, max_time, len(pred_pe)), pred_pe, 'g--')
                ax.plot(np.linspace(0, max_time, len(pred_pe)), pred_pe+pred_ke, 'r--')
            elif len(bodies) == 2:
                if plot_data:
                    te = doublependulum_all_energy(data, constants['masses'], constants['lengths'], g=g)
                    ax.plot(np.linspace(0, max_time_data, len(te)), te, 'r-')
                pred_te = doublependulum_all_energy(predictions, constants['masses'], constants['lengths'], g=g)
                ax.plot(np.linspace(0, max_time, len(pred_te)), pred_te, 'r--')

            else:
                raise ValueError

        if mse:
            subplt += 1
            ax = fig.add_subplot(height, width, subplt)
            if i == 0:
                ax.set_title('Mean square error:')
            error = calculate_error(data, predictions[predictions['time']<=max_time_data],
                                    my_net.settings['feature_names'])
            if period_q != None:
                q = my_net.get_pos_features()
                error = error.abs()
                error[q] = period_q - (error[q] - period_q).abs()
            error = error.values**2
            ax.plot(np.linspace(0, max_time_data, len(error)), error.mean(axis=-1), 'r')

    save_show(fig, save_name)

def predict_pendulum_rand(data_path, store_name, num_runs, nn_paths, **kwargs):
    """Plots the predictions of one or several NNs of one (single/double) pendulum trajectory using a random initial state from data in infinit loop.

    Data and NNs should all share the same scaling!

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        num_runs (int): Number of runs from which to draw.
        nn_paths (str, str[]): Paths to where the NNs are saved.
    """
    while True:
        run_num = random.randint(0, num_runs-1)
        predict_pendulum(data_path, store_name, run_num, nn_paths, **kwargs)
