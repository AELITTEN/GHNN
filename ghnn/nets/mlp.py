import os
import json
from collections import OrderedDict
from itertools import product
from shutil import rmtree
import numpy as np
import pandas as pd
import torch
from ghnn.nets.nnet import NNet
from ghnn.nets.pt_modules import DenseModule
from ghnn.data.adaptor import Data_adaptor

__all__ = ['MLP']

class MLP(NNet):
    """Multilayer perceptron for time-stepped data of Hamiltonian systems with a PyTorch backend.

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
        settings['nn_type'] = 'MLP'
        settings['layer'] = 5
        settings['neurons'] = 128
        return settings

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        for i in range(self.settings['layer']):
            inputs = self.model['dense_'+str(i+1)](inputs)
        outputs = self.model['output'](inputs)
        return outputs

    def create_model(self):
        """Creates the torch ModuleDict from the settings."""
        if not isinstance(self.settings['neurons'], list):
            self.settings['neurons'] = [self.settings['neurons']] * self.settings['layer']
        if not isinstance(self.settings['activations'], list):
            self.settings['activations'] = [self.settings['activations']] * self.settings['layer']
        modules = torch.nn.ModuleDict()
        modules['dense_1'] = DenseModule(self.dim*2,
                                         self.settings['neurons'][0],
                                         self.settings['activations'][0])
        for i in range(1, self.settings['layer']):
            modules['dense_'+str(i+1)] = DenseModule(self.settings['neurons'][i-1],
                                                     self.settings['neurons'][i],
                                                     self.settings['activations'][i])
        modules['output'] = DenseModule(self.settings['neurons'][-1], self.dim*2, None)
        return modules
