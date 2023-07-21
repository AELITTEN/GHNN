"""Some plotting helpers."""
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_2D', 'plot_3D', 'save_show', 'med_quart', 'mean_var']

colors = ['r', 'b', 'g', 'm', 'y', 'c', 'k']

def plot_2D(ax, bodies, data, dims=['x', 'y'], symbol='-', init=True):
    """Plots 2D trajectories in cartesian coordinates.

    Args:
        ax (matplotlib.axes): Axes to plot on.
        bodies (str[]): Identifiers of the bodies for which the trajectories should be plotted.
        data (pd.DataFrame): Data of the trajectories.
        dims (str[]): Which two dimensions should be plotted.
        symbol (str): Symbol used for plotting.
        init (bool): Whehter to plot the initial positions separately or not.
    """
    for i, body in enumerate(bodies):
        ax.plot(data['q_'+body+'_'+dims[0]].values, data['q_'+body+'_'+dims[1]].values, colors[(i%len(colors))]+symbol, label=body)
        if init:
            ax.plot(data['q_'+body+'_'+dims[0]].values[0], data['q_'+body+'_'+dims[1]].values[0], colors[(i%len(colors))]+'o')

def plot_3D(ax, bodies, data, symbol='-', init=True):
    """Plots 3D trajectories in cartesian coordinates.

    Args:
        ax (matplotlib.axes): Axes to plot on.
        bodies (str[]): Identifiers of the bodies for which the trajectories should be plotted.
        data (pd.DataFrame): Data of the trajectories.
        symbol (str): Symbol used for plotting.
        init (bool): Whehter to plot the initial positions separately or not.
    """
    for i, body in enumerate(bodies):
        ax.plot(data['q_'+body+'_x'].values,
                data['q_'+body+'_y'].values,
                data['q_'+body+'_z'].values,
                colors[(i%len(colors))]+symbol)
        if init:
            ax.plot([data['q_'+body+'_x'].values[0]],
                    [data['q_'+body+'_y'].values[0]],
                    [data['q_'+body+'_z'].values[0]],
                    colors[(i%len(colors))]+'o')

def save_show(fig, save_name):
    """Decides whether to show the figure or save it."""
    if save_name:
        fig.savefig(save_name)
    else:
        plt.show()

def med_quart(data):
    """Calculates quatile statistics of a timeseries."""
    med = np.quantile(data, 0.5, axis=0)
    lower = np.quantile(data, 0.25, axis=0)
    upper = np.quantile(data, 0.75, axis=0)
    return med, lower, upper

def mean_var(data):
    """Calculates mean and two-sided variance statistics of a timeseries."""
    mean = data.mean(axis=0)
    pos_var = data - mean
    pos_var[pos_var<0] = 0
    pos_var = pos_var ** 2
    pos_var = pos_var.mean(axis = 0)
    pos_std = pos_var**(1/2)
    neg_var = mean - data
    neg_var[neg_var<0] = 0
    neg_var = neg_var ** 2
    neg_var = neg_var.mean(axis = 0)
    neg_std = neg_var**(1/2)

    return mean, (mean-neg_std), (pos_std+mean)
