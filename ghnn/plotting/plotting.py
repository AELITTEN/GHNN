"""Plotting tools.

Example:
    >>> data_path = '<path-to-data>'
    >>> store_name = 'all_runs.h5.1'
    >>> run_num = 9985
    >>> num_runs = 100
    >>> plot_run(data_path, store_name, run_num)
    >>> plot_num_timesteps_hist(data_path, store_name, num_runs)
    >>> plot_timesteps_hist(data_path, store_name, num_runs)
"""
import os
import math
import random
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ghnn.plotting.helpers import *
from ghnn.analyze import dataset_mse, dataset_mae

__all__ = ['plot_run', 'plot_run_rand', 'plot_pendulum', 'plot_pendulum_rand', 'plot_loss', 'plot_loss_moments', 'plot_data_mse', 'plot_data_mae', 'plot_data_mse_moments', 'plot_data_mae_moments']

def plot_run(data_path, store_name, run_num, save_name=None, max_time=None, d3=False, dims=['x','y'], phase_space=False):
    """Plots the trajectory of all bodies from one specified N-body run in cartesian coordinates.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        run_num (int): Number of the trajectory that is supposed to be plotted.
        save_name (str, path-like, binary file-like): Path where to save the plot.
          If None then the plot is shown and not saved.
        max_time (float): Maximal time until when to plot the trajectory.
        d3 (bool): Switch for a 3D or 2D plot.
        dims (str[]): Which two dimensions should be plotted, in case d3==False.
        phase_space (bool): Whether to plot postions and momenta additionally over time.
    """
    data_path = os.path.join(data_path, store_name)
    data = pd.read_hdf(data_path, '/run' + str(run_num))
    constants = pd.read_hdf(data_path, '/constants')
    if max_time:
        data = data[data['time']<=max_time]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.suptitle(f'Run number {run_num}:')
    if d3:
        if phase_space:
            ax = fig.add_subplot(1, 3, 1, projection='3d')
        else:
            ax = fig.add_subplot(111, projection='3d')
        plot_3D(ax, constants['bodies'], data)
    else:
        if phase_space:
            ax = fig.add_subplot(1, 3, 1)
        else:
            ax = fig.add_subplot(111)
        plot_2D(ax, constants['bodies'], data, dims=dims)
    ax.set_title('Trajectories:')

    if phase_space:
        positions = data[[col for col in data if col[0]=='q']]
        momenta = data[[col for col in data if col[0]=='p']]

        ax = fig.add_subplot(1, 3, 2)
        ax.set_title('Positions:')
        for q in positions:
            ax.plot(data['time'], positions[q], label=q)
        ax.legend()

        ax = fig.add_subplot(1, 3, 3)
        ax.set_title('Momenta:')
        for p in momenta:
            ax.plot(data['time'], momenta[p], label=p)
        ax.legend()

    save_show(fig, save_name)

def plot_run_rand(data_path, store_name, num_runs, **kwargs):
    """Plots the trajectory of all bodies from one random N-body run in cartesian coordinates.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        num_runs (int): Number of runs from which to draw.
    """
    while True:
        run_num = random.randint(0, num_runs-1)
        plot_run(data_path, store_name, run_num, **kwargs)

def plot_pendulum(data_path, store_name, run_num, save_name=None, max_time=None, phase_space=False, period_q=None):
    """Plots the trajectory of a (single/double) pendulum from one specified run.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        run_num (int): Number of the trajectory that is supposed to be plotted.
        save_name (str, path-like, binary file-like): Path where to save the plot.
          If None then the plot is shown and not saved.
        max_time (float): Maximal time until when to plot the trajectory.
        phase_space (bool): Whether to plot postions and momenta additionally over time.
        period_q (float): If not None inputs are mapped to [-period_q, period_q].
    """
    data_path = os.path.join(data_path, store_name)
    data = pd.read_hdf(data_path, '/run' + str(run_num))
    constants = pd.read_hdf(data_path, '/constants')
    if max_time:
        data = data[data['time']<=max_time]
    if period_q != None:
        q = [col for col in data if col[0]=='q']
        data[q] = (data[q] + period_q) % (2 * period_q) - period_q
    bodies = constants['bodies']

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.suptitle(f'Run number {run_num}:')

    if phase_space:
        colors = ['r', 'b', 'g', 'm', 'y', 'c', 'k']
        positions = data[[col for col in data if col[0]=='q']]
        momenta = data[[col for col in data if col[0]=='p']]

        ax = fig.add_subplot(1, 3, 2)
        ax.set_title('Positions:')
        for j, q in enumerate(positions):
            k_old = 0
            for k, _ in enumerate(data.loc[1:,'time'], start=1):
                if abs(positions.loc[k-1, q] - positions.loc[k, q]) > 4:
                    ax.plot(data.loc[k_old:k-1, 'time'], positions.loc[k_old:k-1, q], colors[(j%len(colors))]+'-')
                    k_old = k
            ax.plot(data.loc[k_old:, 'time'], positions.loc[k_old:, q], colors[(j%len(colors))]+'-', label=q)
        ax.legend()

        ax = fig.add_subplot(1, 3, 3)
        ax.set_title('Momenta:')
        for j, p in enumerate(momenta):
            ax.plot(data['time'], momenta[p], colors[(j%len(colors))]+'-', label=p)
        ax.legend()

        ax = fig.add_subplot(1, 3, 1)
    else:
        ax = fig.add_subplot(111)
    ax.set_title('Trajectories:')

    if len(bodies) == 1:
        data['q_'+bodies[0]+'_x'] = constants['length'] * np.sin(data['q_'+bodies[0]])
        data['q_'+bodies[0]+'_y'] = - constants['length'] * np.cos(data['q_'+bodies[0]])
        ax.plot([0, data['q_'+bodies[0]+'_x'].values[0]], [0, data['q_'+bodies[0]+'_y'].values[0]], 'k-')
    elif len(bodies) == 2:
        data['q_'+bodies[0]+'_x'] = constants['lengths'][0] * np.sin(data['q_'+bodies[0]])
        data['q_'+bodies[0]+'_y'] = - constants['lengths'][0] * np.cos(data['q_'+bodies[0]])
        data['q_'+bodies[1]+'_x'] = constants['lengths'][1] * np.sin(data['q_'+bodies[1]]) + data['q_'+bodies[0]+'_x']
        data['q_'+bodies[1]+'_y'] = - constants['lengths'][1] * np.cos(data['q_'+bodies[1]]) + data['q_'+bodies[0]+'_y']
        ax.plot([0, data['q_'+bodies[0]+'_x'].values[0]], [0, data['q_'+bodies[0]+'_y'].values[0]], 'k-')
        ax.plot([data['q_'+bodies[0]+'_x'].values[0], data['q_'+bodies[1]+'_x'].values[0]],
                [data['q_'+bodies[0]+'_y'].values[0], data['q_'+bodies[1]+'_y'].values[0]], 'k-')
    else:
        raise ValueError

    plot_2D(ax, bodies, data)

    save_show(fig, save_name)

def plot_pendulum_rand(data_path, store_name, num_runs, **kwargs):
    """Plots the trajectory of a (single/double) pendulum from one random run.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        num_runs (int): Number of runs from which to draw.
    """
    while True:
        run_num = random.randint(0, num_runs-1)
        plot_pendulum(data_path, store_name, run_num, **kwargs)

def plot_loss(nn_paths, rol_avg=1, one_plot=False, save_name=None):
    """Plots the training and validation loss of (possibly multiple) NNs.

    Can use a rolling average to smooth the loss.

    Args:
        nn_paths (str[], str, path-like object): Path where the NNs are saved.
        rol_avg (int): Number of datapoints for the rolling average.
        one_plot (bool): If true the validation loss of all NNs is combined in one plot.
        save_name (str, path-like, binary file-like): Path where to save the plot.
          If None then the plot is shown and not saved.
    """
    if not isinstance(nn_paths, list):
        nn_paths = [nn_paths]

    if not one_plot:
        height = math.floor(math.sqrt(len(nn_paths)))
        width = math.ceil(len(nn_paths)/height)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    if one_plot:
        ax = fig.add_subplot(111)
    for i, nn_path in enumerate(nn_paths, start=1):
        data = np.loadtxt(os.path.join(nn_path, 'logs', 'loss.txt'))
        loss = np.cumsum(data[:,1:], axis=0)
        loss[rol_avg:,:] = loss[rol_avg:,:] - loss[:-rol_avg,:]
        loss = loss[rol_avg-1:] / rol_avg
        path = os.path.normpath(nn_path).split(os.path.sep)
        if path[-1][:2] == 'nn':
            name = path[-2]
        else:
            name = path[-1]

        if not one_plot:
            ax = fig.add_subplot(height, width, i)
            ax.set_title(name)

            ax.plot(data[rol_avg-1:,0].astype(int), loss[:,0], label='Training loss')
            ax.plot(data[rol_avg-1:,0].astype(int), loss[:,1], label='Validation loss')

            ax.set_yscale('log')
            ax.legend()
        else:
            ax.plot(data[rol_avg-1:,0].astype(int), loss[:,1], label=name)

    if one_plot:
        ax.set_yscale('log')
        ax.legend()

    save_show(fig, save_name)

def plot_loss_moments(nn_paths_list, stat, rol_avg=1, one_plot=False, save_name=None):
    """Plots the moments of training and validation loss of multiple NNs.

    Can use a rolling average to smooth the loss.

    Args:
        nn_paths_list (str[], str[][]): Paths where the NNs are saved.
        stat (str): Type of statistic.
        rol_avg (int): Number of datapoints for the rolling average.
        one_plot (bool): If true the validation loss of all NNs is combined in one plot.
        save_name (str, path-like, binary file-like): Path where to save the plot.
          If None then the plot is shown and not saved.
    """
    if not isinstance(nn_paths_list[0], list):
        nn_paths_list = [nn_paths_list]

    if not one_plot:
        height = math.floor(math.sqrt(len(nn_paths_list)))
        width = math.ceil(len(nn_paths_list)/height)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    if one_plot:
        ax = fig.add_subplot(111)
    for i, nn_paths in enumerate(nn_paths_list, start=1):
        losses = []
        for nn_path in nn_paths:
            data = np.loadtxt(os.path.join(nn_path, 'logs', 'loss.txt'))
            loss = np.cumsum(data[:,1:], axis=0)
            loss[rol_avg:,:] = loss[rol_avg:,:] - loss[:-rol_avg,:]
            loss = loss[rol_avg-1:] / rol_avg
            losses.append(loss)
        losses = np.array(losses)

        if stat == 'med_quart':
            m, l, u = med_quart(losses)
        elif stat == 'mean_var':
            m, l, u = mean_var(losses)
        else:
            raise ValueError('Use either "med_quart" or "mean_var" for stat')

        path = os.path.normpath(nn_path).split(os.path.sep)
        if path[-1][:2] == 'nn':
            name = path[-2]
        else:
            name = path[-1]

        if not one_plot:
            ax = fig.add_subplot(height, width, i)
            ax.set_title(name)

            ax.plot(data[rol_avg-1:,0].astype(int), m[:,0], label='Training loss')
            ax.fill_between(data[rol_avg-1:,0].astype(int), l[:,0], u[:,0], alpha=0.2)
            ax.plot(data[rol_avg-1:,0].astype(int), m[:,1], label='Validation loss')
            ax.fill_between(data[rol_avg-1:,0].astype(int), l[:,1], u[:,1], alpha=0.2)

            ax.set_yscale('log')
            ax.legend()
        else:
            ax.plot(data[rol_avg-1:,0].astype(int), m[:,1], label=name)
            ax.fill_between(data[rol_avg-1:,0].astype(int), l[:,1], u[:,1], alpha=0.2)

    if one_plot:
        ax.set_yscale('log')
        ax.legend()

    save_show(fig, save_name)

def plot_data_mse(data_path, store_name, nn_paths, save_name=None, **kwargs):
    """Plots the mean of the MSE at each timestep for all trajectories predicted by (possibly) multiple NNs.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        nn_paths (str[], str, path-like object): Path where the NNs are saved.
        save_name (str, path-like, binary file-like): Path where to save the plot.
          If None then the plot is shown and not saved.
    """
    if not isinstance(nn_paths, list):
        nn_paths = [nn_paths]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot()
    for nn_path in nn_paths:
        mse = dataset_mse(data_path, store_name, nn_path, **kwargs)
        path = os.path.normpath(nn_path).split(os.path.sep)
        if path[-1][:2] == 'nn':
            name = path[-2]
        else:
            name = path[-1]
        ax.plot(mse['time'], mse['mse'], label=name)
    ax.legend()

    save_show(fig, save_name)

def plot_data_mae(data_path, store_name, nn_paths, save_name=None, **kwargs):
    """Plots the mean of the MAE at each timestep for all trajectories predicted by (possibly) multiple NNs.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        nn_paths (str[], str, path-like object): Path where the NNs are saved.
        save_name (str, path-like, binary file-like): Path where to save the plot.
          If None then the plot is shown and not saved.
    """
    if not isinstance(nn_paths, list):
        nn_paths = [nn_paths]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot()
    for nn_path in nn_paths:
        mae = dataset_mae(data_path, store_name, nn_path, **kwargs)
        path = os.path.normpath(nn_path).split(os.path.sep)
        if path[-1][:2] == 'nn':
            name = path[-2]
        else:
            name = path[-1]
        ax.plot(mae['time'], mae['mae'], label=name)
    ax.legend()

    save_show(fig, save_name)

def plot_data_mse_moments(data_path, store_name, nn_paths_list, stat, save_name=None, **kwargs):
    """Plots the moments of the mean of the MSE at each timestep for all trajectories predicted by multiple NNs.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        nn_paths_list (str[], str[][]): Paths where the NNs are saved.
        stat (str): Type of statistic.
        save_name (str, path-like, binary file-like): Path where to save the plot.
          If None then the plot is shown and not saved.
    """
    if not isinstance(nn_paths_list[0], list):
        nn_paths_list = [nn_paths_list]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot()
    for nn_paths in nn_paths_list:
        mses = []
        for nn_path in nn_paths:
            mse = dataset_mse(data_path, store_name, nn_path, **kwargs)
            mses.append(mse['mse'])
        mses = np.array(mses)

        if stat == 'med_quart':
            m, l, u = med_quart(mses)
        elif stat == 'mean_var':
            m, l, u = mean_var(mses)
        else:
            raise ValueError('Use either "med_quart" or "mean_var" for stat')

        path = os.path.normpath(nn_path).split(os.path.sep)
        if path[-1][:2] == 'nn':
            name = path[-2]
        else:
            name = path[-1]

        ax.plot(mse['time'], m, label=name)
        ax.fill_between(mse['time'], l, u, alpha=0.2)
    ax.legend()

    save_show(fig, save_name)

def plot_data_mae_moments(data_path, store_name, nn_paths_list, stat, save_name=None, **kwargs):
    """Plots the moments of the mean of the MAE at each timestep for all trajectories predicted by multiple NNs.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        nn_paths_list (str[], str[][]): Paths where the NNs are saved.
        stat (str): Type of statistic.
        save_name (str, path-like, binary file-like): Path where to save the plot.
          If None then the plot is shown and not saved.
    """
    if not isinstance(nn_paths_list[0], list):
        nn_paths_list = [nn_paths_list]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot()
    for nn_paths in nn_paths_list:
        maes = []
        for nn_path in nn_paths:
            mae = dataset_mae(data_path, store_name, nn_path, **kwargs)
            maes.append(mae['mae'])
        maes = np.array(maes)

        if stat == 'med_quart':
            m, l, u = med_quart(maes)
        elif stat == 'mean_var':
            m, l, u = mean_var(maes)
        else:
            raise ValueError('Use either "med_quart" or "mean_var" for stat')

        path = os.path.normpath(nn_path).split(os.path.sep)
        if path[-1][:2] == 'nn':
            name = path[-2]
        else:
            name = path[-1]

        ax.plot(mae['time'], m, label=name)
        ax.fill_between(mae['time'], l, u, alpha=0.2)
    ax.legend()

    save_show(fig, save_name)
