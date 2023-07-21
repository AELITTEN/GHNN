import math
import numpy as np
import pandas as pd
import torch
from ghnn.nets.nnet import NNet
from ghnn.nets.pt_modules import Double_HenonModule

__all__ = ['Double_HenonNet']

class Double_HenonNet(NNet):
    """Double HenonNet for data of Hamiltonian systems with a PyTorch backend.

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
        settings['nn_type'] = 'double_HenonNet'
        settings['modules'] = 3
        settings['units1'] = 25
        settings['units2'] = 25
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
        if not isinstance(self.settings['units1'], list):
            self.settings['units1'] = [self.settings['units1']] * self.settings['modules']
        if not isinstance(self.settings['units2'], list):
            self.settings['units2'] = [self.settings['units2']] * self.settings['modules']
        if not isinstance(self.settings['activations'], list):
            self.settings['activations'] = [self.settings['activations']] * self.settings['modules']
        modules = torch.nn.ModuleDict()
        for i in range(self.settings['modules']):
            modules['henon_'+str(i+1)] = Double_HenonModule(self.dim, self.settings['units1'][i], self.settings['units2'][i], self.settings['activations'][i])
        return modules
