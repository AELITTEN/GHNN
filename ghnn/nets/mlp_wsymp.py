import os
import json
from collections import OrderedDict
from itertools import product
from shutil import rmtree
import numpy as np
import pandas as pd
import torch
from torch.func import vmap, vjp
from ghnn.nets.nnet import NNet
from ghnn.nets.pt_modules import DenseModule
from ghnn.data.adaptor import Data_adaptor

__all__ = ['MLP_wsymp']

class MLP_wsymp(NNet):
    """Multilayer perceptron for time-stepped data of Hamiltonian systems with a PyTorch backend.

    Trained with an added symplectic constraint in the loss.

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
        settings['nn_type'] = 'MLP_wsymp'
        settings['layer'] = 5
        settings['neurons'] = 128
        settings['p_range'] = [-2, 2]
        settings['q_range'] = [-3.1416, 3.1416]
        settings['p_steps'] = 10
        settings['q_steps'] = 10
        settings['symp_lambda'] = 1
        return settings

    def check_settings(self):
        """Checks the settings for anything that causes problems."""
        super().check_settings()
        if self.settings['loss_weights']:
            print('Loss weights are ignored in these NNs!')
        dim = int(len(self.settings['feature_names']) / 2)
        if not isinstance(self.settings['p_range'][0], list):
            self.settings['p_range'][0] = [self.settings['p_range'][0]] * dim
        if not isinstance(self.settings['p_range'][1], list):
            self.settings['p_range'][1] = [self.settings['p_range'][1]] * dim
        if not isinstance(self.settings['q_range'][0], list):
            self.settings['q_range'][0] = [self.settings['q_range'][0]] * dim
        if not isinstance(self.settings['q_range'][1], list):
            self.settings['q_range'][1] = [self.settings['q_range'][1]] * dim

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

    def load_data(self, data_path=None):
        """Loads the training data.

        Args:
            data_path (str, path-like object, pandas.HDFStore or file-like object):
              Path to the HDF5 training data file.
        Returns:
            2-element tuple:
            - (*Data_adaptor*): The custom Data_adaptor with features and labels.
            - (*torch.Tensor*): The coordinates for the symplectic residual loss.
        """
        data = super().load_data(data_path=data_path)
        p_range = self.settings['p_range']
        q_range = self.settings['q_range']
        ps = np.linspace(p_range[0], p_range[1], self.settings['p_steps'], axis=1) / data.mom_scaling
        qs = np.linspace(q_range[0], q_range[1], self.settings['q_steps'], axis=1) / data.pos_scaling
        x = [p for p in ps] + [q for q in qs]
        x = np.meshgrid(*x, indexing='ij')
        coordinates = np.stack(x, axis=-1).reshape((-1, 2*self.dim))
        coordinates = torch.tensor(coordinates, dtype=self.dtype, device=self.device)
        return data, coordinates

    def my_loss(self, train_features, train_labels):
        """Calculates the loss."""
        return self.loss(self, train_features, train_labels)

    def jacobian(self, inputs):
        """Calculates the jacobian of the NN with regards to its inputs"""
        num_in = inputs.shape[0]
        inputs.requires_grad_()
        unit_vectors = np.eye(2*self.dim)
        unit_vectors = unit_vectors.repeat(num_in, 0)
        unit_vectors = unit_vectors.reshape(2*self.dim, num_in, 2*self.dim)
        unit_vectors = torch.tensor(unit_vectors, dtype=self.dtype, device=self.device)
        _, vjp_fn = vjp(self.forward, inputs)
        jacobian, = vmap(vjp_fn)(unit_vectors)
        return jacobian

    def calc_symp_mse(self, pos):
        jac = self.jacobian(pos)
        jac = torch.swapaxes(jac, 0, 1)
        dim = self.dim
        up = np.concatenate((np.zeros((dim, dim)), np.eye(dim)), axis=1)
        down = np.concatenate((-np.eye(dim), np.zeros((dim, dim))), axis=1)
        J = np.concatenate((up, down), axis=0)
        J = torch.tensor(J, dtype=self.dtype, device=self.device)
        symp_loss = torch.swapaxes(jac, 1, 2) @ J @ jac - J
        symp_loss = torch.mean(symp_loss**2, axis=(1,2))
        return symp_loss

    def calc_symp_mae(self, pos):
        jac = self.jacobian(pos)
        jac = torch.swapaxes(jac, 0, 1)
        dim = self.dim
        up = np.concatenate((np.zeros((dim, dim)), np.eye(dim)), axis=1)
        down = np.concatenate((-np.eye(dim), np.zeros((dim, dim))), axis=1)
        J = np.concatenate((up, down), axis=0)
        J = torch.tensor(J, dtype=self.dtype, device=self.device)
        symp_loss = torch.swapaxes(jac, 1, 2) @ J @ jac - J
        symp_loss = torch.mean(torch.abs(symp_loss), axis=(1,2))
        return symp_loss
