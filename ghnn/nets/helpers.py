"""A few helpers for the NNs."""
import os
import json
import pandas as pd
from ghnn.nets.mlp import MLP
from ghnn.nets.la_sympnet import LA_SympNet
from ghnn.nets.g_sympnet import G_SympNet
from ghnn.nets.ghnn import GHNN
from ghnn.nets.henonnet import HenonNet
from ghnn.nets.double_henonnet import Double_HenonNet

__all__ = ['net_from_dir', 'params_from_settings', 'params_from_net', 'predict_trajectories']

def _load_settings(path):
    if path[-13:] == 'settings.json':
        settings_path = path
        path = path[:-13]
        if not path:
            path = '.'
    else:
        settings_path = os.path.join(path, 'settings.json')

    with open(settings_path) as file_:
        settings = json.load(file_)

    return settings

def net_from_dir(path, device=None):
    """Initializes the correct kind of NN from a settings file.

    Args:
        path (str): Path where to find a settings file for the NN.
        device (str): The device to load the NN on. 'cpu' or 'gpu'.
          If None settings['device'] is used.

    Returns:
        NNet: The correct subclass of NNet.
    """
    settings = _load_settings(path)
    if settings['nn_type'] == 'MLP':
        my_net = MLP(path)
    elif settings['nn_type'] == 'LA_SympNet':
        my_net = LA_SympNet(path)
    elif settings['nn_type'] == 'G_SympNet':
        my_net = G_SympNet(path)
    elif settings['nn_type'] == 'GHNN':
        my_net = GHNN(path, device=device)
    elif settings['nn_type'] == 'HenonNet':
        my_net = HenonNet(path, device=device)
    elif settings['nn_type'] == 'double_HenonNet':
        my_net = Double_HenonNet(path, device=device)
    else:
        raise ValueError('No known nn_type could be identified. '
                         'Use "MLP", "LA_SympNet", "G_SympNet", '
                         '"GHNN", "HenonNet" or "double_HenonNet"')
    return my_net

def params_from_settings(path):
    """Calculates the number of trainable parameters in an NN from the settings.

    Args:
        path (str): Path where to find a settings file for the NN.

    Returns:
        int tuple: The number of trainable parameters and the effective number of free parameters.
    """
    settings = _load_settings(path)
    params = 0
    real_params = 0

    if  settings['nn_type'] == 'MLP':
        if not isinstance(settings['neurons'], list):
            settings['neurons'] = [settings['neurons']] * settings['layer']
        params += (len(settings['feature_names'])+1) * settings['neurons'][0]
        for i in range(1, settings['layer']):
            params += (settings['neurons'][i-1] + 1) * settings['neurons'][i]
        params += (settings['neurons'][-1] + 1) * len(settings['label_names'])
        real_params = params

    elif settings['nn_type'] == 'LA_SympNet':
        dim = int(len(settings['feature_names'])/2)
        if not isinstance(settings['sublayer'], list):
            settings['sublayer'] = [settings['sublayer']] * settings['layer']
        for sublayer in settings['sublayer']:
            params += (sublayer * dim * 2 + 3) * dim
            real_params += (sublayer * (dim+1)/2 + 3) * dim
        params += (settings['sublayer'][-1] * dim * 2 + 2) * dim
        real_params += (settings['sublayer'][-1] * (dim+1)/2 + 2) * dim

    elif settings['nn_type'] == 'G_SympNet':
        dim = int(len(settings['feature_names'])/2)
        if not isinstance(settings['units'], list):
            settings['units'] = [settings['units']] * settings['layer']
        for units in settings['units']:
            params += units * (dim + 2)
        real_params = params

    return params, real_params

def params_from_net(my_net):
    """Calculates the number of trainable parameters in a given NN.

    Args:
        my_net (NNet): The NN of which to extract the number of trainable parameters.

    Returns:
        int: The number of trainable parameters.
    """
    params = sum(p.numel() for p in my_net.parameters() if p.requires_grad)

    return params

def predict_trajectories(nn, inp, max_time, **kwargs):
    """A small helper to unify prediction of several trajectories with different NN types.

    Args:
        nn (ghnn.nets.NNet): The NN that does the prediction.
        inp (pd.DataFrame, pd.Series): The initial conditions for the trajectories.
        max_time (float, float[]): The final time until when to predict.
          Possibly different per trajctory.

    Returns:
        pd.DataFrame: The predicted trajectories.
    """
    if not isinstance(max_time, list):
        max_time = [max_time]
    end_time = max(max_time)
    predictions = nn.predict_path(inp[nn.settings['feature_names']].values, end_time, **kwargs)
    if isinstance(inp, pd.DataFrame):
        index = pd.MultiIndex.from_product([inp.index, predictions.loc[0].index],
                                            names=["run", "timestep"])
        predictions.index = index
    runs = predictions.index.get_level_values(0).unique()
    for i, run in enumerate(runs):
        steps = int(max_time[i] / nn.settings['step_size'])
        predictions.loc[run] = predictions.loc[(run, slice(0,steps)),:]
    predictions.dropna(how='all', inplace=True)
    if isinstance(inp, pd.Series):
        predictions.index = predictions.loc[0].index

    return predictions
