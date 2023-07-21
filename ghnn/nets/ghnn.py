import os
import subprocess
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from ghnn.nets.nnet import NNet
from ghnn.nets.pt_modules import SE_HamiltonModule, Double_SE_HamiltonModule, SV_HamiltonModule, Double_SV_HamiltonModule, HamiltonModule

__all__ = ['GHNN']

class GHNN(NNet):
    """Generalized Hamiltonian NN for data of Hamiltonian systems with a PyTorch backend.

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
        settings['nn_type'] = 'GHNN'
        settings['l_hamilt'] = 5
        settings['layer'] = 2
        settings['neurons'] = 25
        settings['integrators'] = 'symp_euler'
        return settings

    def forward(self, inputs):
        """Defines the computation performed at every call."""
        if inputs.size(-1) == self.dim*2:
            p, q = inputs[..., :self.dim], inputs[..., self.dim:]
        else:
            raise ValueError
        for i in range(self.settings['l_hamilt']):
            p, q = self.model[f'hamilton_{i+1}']([p, q])
        return torch.cat([p, q], dim=-1)

    def create_model(self):
        if not isinstance(self.settings['layer'], list):
            self.settings['layer'] = [self.settings['layer']] * self.settings['l_hamilt']
        if not isinstance(self.settings['neurons'], list):
            self.settings['neurons'] = [self.settings['neurons']] * self.settings['l_hamilt']
        if not isinstance(self.settings['activations'], list):
            self.settings['activations'] = [self.settings['activations']] * self.settings['l_hamilt']
        if not isinstance(self.settings['integrators'], list):
            self.settings['integrators'] = [self.settings['integrators']] * self.settings['l_hamilt']

        modules = torch.nn.ModuleDict()
        for i in range(self.settings['l_hamilt']):
            layer = self.settings['layer'][i]
            integrator = self.settings['integrators'][i]
            if layer == 1 and (integrator == 'symp_euler' or integrator == 'leapfrog'):
                modules[f'hamilton_{i+1}'] = SE_HamiltonModule(self.dim,
                                                               self.settings['neurons'][i],
                                                               self.settings['activations'][i])
            elif layer == 1 and integrator == 'stoermer_verlet':
                modules[f'hamilton_{i+1}'] = SV_HamiltonModule(self.dim,
                                                               self.settings['neurons'][i],
                                                               self.settings['activations'][i])
            elif layer == 2 and (integrator == 'symp_euler' or integrator == 'leapfrog'):
                modules[f'hamilton_{i+1}'] = Double_SE_HamiltonModule(self.dim,
                                                                      self.settings['neurons'][i],
                                                                      self.settings['activations'][i])
            elif layer == 2 and integrator == 'stoermer_verlet':
                modules[f'hamilton_{i+1}'] = Double_SV_HamiltonModule(self.dim,
                                                                      self.settings['neurons'][i],
                                                                      self.settings['activations'][i])
            else:
                modules[f'hamilton_{i+1}'] = HamiltonModule(self.dim,
                                                            self.settings['neurons'][i],
                                                            self.settings['layer'][i],
                                                            self.settings['activations'][i],
                                                            self.settings['integrators'][i])
        return modules

    def to_dict(self):
        """Transforms this class to a json dict.

        Returns:
            dict: The weights and settings in one json dict.
        """
        nn_dict = OrderedDict()
        nn_dict['nn_type'] = self.settings['nn_type']
        nn_dict['feature_names'] = self.settings['feature_names']
        nn_dict['label_names'] = self.settings['label_names']
        git_dir = os.path.dirname(__file__)[:-9] + '.git'
        version = subprocess.check_output(['git',
                                           '--git-dir=' + git_dir,
                                           'rev-parse', 'HEAD']).decode('ascii').strip()
        nn_dict['code_version'] = version

        weight_dict = {}
        weight_dict['input'] = []
        for mod in self.model:
            if isinstance(self.model[mod], HamiltonModule):
                int_dict = {}
                for int_mod in self.model[mod].model:
                    l = []
                    for param in self.model[mod].model[int_mod].params:
                        l.append(self.model[mod].model[int_mod].params[param].detach().tolist())
                    int_dict[int_mod] = l
                weight_dict[mod] = int_dict
            else:
                l = []
                for param in self.model[mod].params:
                    l.append(self.model[mod].params[param].detach().tolist())
                weight_dict[mod] = l
        nn_dict.update({'weights': weight_dict, '_parsed_settings': self.settings})
        return nn_dict

    def load_from_json(self, path='nn.json', device=None):
        """Loads the NN from a json file.

        The attributes settings and model are overwritten with the data from the json file.

        Args:
            path (str, path-like object): Path where to find the json file.
            device (str): The device to load the NN on. 'cpu' or 'gpu'.
              If None settings['device'] is used.
        """
        with open(path, 'r') as file_:
            nn_dict = json.load(file_)
        self.settings.update(nn_dict['_parsed_settings'])
        if device:
            self.settings['device'] = device
        self.check_settings()
        self.dim = int(len(self.settings['feature_names']) / 2)
        self.model = self.create_model()
        weight_dict = nn_dict['weights']
        for mod in self.model:
            if isinstance(self.model[mod], HamiltonModule):
                for int_mod in self.model[mod].model:
                    for i, param in enumerate(self.model[mod].model[int_mod].params):
                        self.model[mod].model[int_mod].params[param] = torch.nn.Parameter(torch.FloatTensor(weight_dict[mod][int_mod][i]).requires_grad_(True))
            else:
                for i, param in enumerate(self.model[mod].params):
                    self.model[mod].params[param] = torch.nn.Parameter(torch.FloatTensor(weight_dict[mod][i]).requires_grad_(True))

    def load_from_checkpoint(self, path):
        """Loads NN weights from a checkpoint.

        Args:
            path (str, path-like object): Path to the exact checkpoint.
        """
        with open(path, 'r') as file_:
            nn_dict = json.load(file_)
        weight_dict = nn_dict['weights']
        for mod in self.model:
            if isinstance(self.model[mod], HamiltonModule):
                for int_mod in self.model[mod].model:
                    for i, param in enumerate(self.model[mod].model[int_mod].params):
                        self.model[mod].model[int_mod].params[param] = torch.nn.Parameter(torch.FloatTensor(weight_dict[mod][int_mod][i]).requires_grad_(True))
            else:
                for i, param in enumerate(self.model[mod].params):
                    self.model[mod].params[param] = torch.nn.Parameter(torch.FloatTensor(weight_dict[mod][i]).requires_grad_(True))
        self.device = self.settings['device']
