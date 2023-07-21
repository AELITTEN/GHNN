import math
import numpy as np
import pandas as pd
import torch
from ghnn.nets.nnet import NNet
from ghnn.nets.pt_modules import HenonModule

__all__ = ['HenonNet']

class HenonNet(NNet):
    """HenonNet for data of Hamiltonian systems with a PyTorch backend.

    Args:
        path (str, path-like object): Path where to find a settings file for the NN.

    Attributes:
        settings (dict): All settings.
        model (torch.nn.ModuleDict): The PyTorch modules of the NN.
        dim (int): The number of spatial features and labels.
        dtype (str): Data type for features and labels. 'float' or 'double'.
        device (str): The device to do the computations. 'cpu' or 'gpu'.
    """

    def default_settings(self):
        settings = super().default_settings()
        settings['nn_type'] = 'HenonNet'
        settings['modules'] = 5
        settings['units'] = 50
        return settings

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        if inputs.size(-1) == self.dim*2:
            p, q = inputs[..., :self.dim], inputs[..., self.dim:]
        else:
            raise ValueError
        for i in range(self.settings['modules']):
            HenonM = self.model['henon_'+str(i+1)]
            p, q = HenonM([p, q])
        return torch.cat([p, q], dim=-1)

    def create_model(self):
        if not isinstance(self.settings['units'], list):
            self.settings['units'] = [self.settings['units']] * self.settings['modules']
        if not isinstance(self.settings['activations'], list):
            self.settings['activations'] = [self.settings['activations']] * self.settings['modules']
        modules = torch.nn.ModuleDict()
        for i in range(self.settings['modules']):
            modules['henon_'+str(i+1)] = HenonModule(self.dim, self.settings['units'][i], self.settings['activations'][i])
        return modules
