import os
import subprocess
import json
from collections import OrderedDict
from itertools import product
from shutil import rmtree
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
from torch.func import vmap, vjp
from abc import ABC, abstractmethod
from ghnn.nets.pt_modules import Module, mse_w_loss, mae_w_loss, mse_symp_loss, mae_symp_loss
from ghnn.data.adaptor import Data_adaptor

__all__ = ['NNet']

class NNet(Module, ABC):
    """Basic abstract NN class for data of Hamiltonian systems with a PyTorch backend.

    Args:
        path (str, path-like object): Path where to find a settings file for the NN.
        device (str): The device to load the NN on. 'cpu' or 'gpu'.
          If None settings['device'] is used.

    Attributes:
        settings (dict): All settings.
        model (torch.nn.ModuleDict): The PyTorch modules of the NN.
        dim (int): The number of spatial features and labels.
        dtype (str): Data type for features and labels. 'float' or 'double'.
        device (str): The device to do the computations. 'cpu' or 'gpu'.
    """

    def __init__(self, path='.', device=None):
        super().__init__()
        model_file = None
        settings_file = None
        if os.path.isdir(path):
            if os.path.isfile(os.path.join(path, 'nn.json')):
                model_file = os.path.join(path, 'nn.json')
            elif os.path.isfile(os.path.join(path, 'settings.json')):
                settings_file = os.path.join(path, 'settings.json')
        elif os.path.isfile(path):
            if path[-7:] == 'nn.json':
                model_file = path
            elif path[-13:] == 'settings.json':
                settings_file = path

        # Load settings
        self.settings = self.default_settings()
        if settings_file is not None:
            with open(settings_file) as file_:
                settings = json.load(file_)
                self.settings.update(settings)
        if model_file is not None:
            self.load_from_json(model_file, device=device)
        else:
            # Settings sanity check
            if device:
                self.settings['device'] = device
            self.check_settings()
            self.dim = int(len(self.settings['feature_names']) / 2)
            if self.settings['seed'] != None:
                torch.manual_seed(self.settings['seed'])
                np.random.seed(self.settings['seed'])
            self.model = self.create_model()

        self.dtype = self.settings['dtype']
        self.device = self.settings['device']

    def default_settings(self):
        """Returns a dict with the default settings."""
        settings = {
            'device': 'cpu',
            'seed': None,
            # Data
            'data_path': 'h_01_training.h5.1',
            'feature_names': ['q_A_x', 'q_A_y', 'q_B_x', 'q_B_y', 'q_C_x', 'q_C_y', 'p_A_x', 'p_A_y', 'p_B_x', 'p_B_y', 'p_C_x', 'p_C_y'],
            'label_names': ['q_A_x', 'q_A_y', 'q_B_x', 'q_B_y', 'q_C_x', 'q_C_y', 'p_A_x', 'p_A_y', 'p_B_x', 'p_B_y', 'p_C_x', 'p_C_y'],
            'max_time': None,
            't_in_T': False,
            'data_frac': 1,
            'dtype': 'float',
            # Callbacks
            'path_logs': 'logs',
            'checkpoint_period': 100,
            # NN
            'nn_type': '',
            'activations': 'sigmoid',
            # Training
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'lr_scheduler': None,
            'loss': 'mse',
            'loss_weights': False,
            'initial_epoch': 0,
            'max_epochs': 25000,
            'batch_size': 200,
            'period_q': None,
            # Model
            'model_path': '.',
            'mom_scaling': 1,
            'pos_scaling': 1
        }
        return settings

    def check_settings(self):
        """Checks the settings for anything that causes problems."""
        # TODO: a lot more sanity checks throwing exceptions and logging
        if 'nn_type' == '':
            print('Something is weird')
        if not torch.cuda.is_available():
            self.settings['device'] = 'cpu'
        elif self.settings['device'][:3] == 'gpu' and len(self.settings['device']) > 3:
            if torch.cuda.device_count() <= int(self.settings['device'][3:]):
                self.settings['device'] = 'gpu' + str(torch.cuda.device_count()-1)
        if 'bodies' in self.settings and 'dims' in self.settings:
            names = product(['q', 'p'], self.settings['bodies'], self.settings['dims'])
            names = [qp+'_'+body+'_'+dim for qp,body,dim in names]
            self.settings['feature_names'] = names
            self.settings['label_names'] = names
        if self.settings['feature_names'] != self.settings['label_names']:
            raise ValueError('Feature and label names should be the same.')
        if len(self.settings['feature_names']) % 2 != 0:
            raise ValueError('Number of features and labels should be even.')
        if 'time' in self.settings['feature_names']:
            print('Something is weird')
        if 'step_size' not in self.settings:
            constants = pd.read_hdf(self.settings['data_path'], '/constants')
            self.settings['step_size'] = constants['step_size']

    @abstractmethod
    def create_model(self):
        """Creates the torch ModuleDict from the settings."""
        pass

    def get_pos_features(self):
        """Returns a list of the position feature names."""
        return [feat for feat in self.settings['feature_names'] if feat[0] == 'q']

    def get_pos_labels(self):
        """Returns a list of the position label names."""
        return [lab for lab in self.settings['label_names'] if lab[0] == 'q']

    def get_mom_features(self):
        """Returns a list of the momentum feature names."""
        return [feat for feat in self.settings['feature_names'] if feat[0] == 'p']

    def get_mom_labels(self):
        """Returns a list of the momentum label names."""
        return [lab for lab in self.settings['label_names'] if lab[0] == 'p']

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

    def load_data(self, data_path=None):
        """Loads the training data.

        Args:
            data_path (str, path-like object, pandas.HDFStore or file-like object):
              Path to the HDF5 training data file.
        Returns:
            Data_adaptor: The custom Data_adaptor with features and labels.
        """
        if not data_path:
            data_path = self.settings['data_path']
        data = Data_adaptor(data_path,
                            self.settings['max_time'],
                            self.settings['data_frac'],
                            self.settings['feature_names'],
                            self.settings['label_names'],
                            self.settings['device'],
                            self.settings['dtype'])
        return data

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
            l = []
            for param in self.model[mod].params:
                l.append(self.model[mod].params[param].detach().tolist())
            weight_dict[mod] = l
        nn_dict.update({'weights': weight_dict, '_parsed_settings': self.settings})
        return nn_dict

    def save_to_json(self, model_path=None, file_name='nn.json'):
        """Saves this object to a json file.

        Args:
            model_path (str, path-like object): Path where to save the data.
              If None, settings['model_path'] is used.
            file_name (str, path-like object): Name of the json file.
        """
        dct = self.to_dict()
        if model_path is None:
            model_path = self.settings['model_path']
        with open(os.path.join(model_path, file_name), 'w') as file_:
            json.dump(dct, file_, indent=4, separators=(',', ': '))

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
            for i, param in enumerate(self.model[mod].params):
                self.model[mod].params[param] = torch.nn.Parameter(torch.FloatTensor(weight_dict[mod][i]).requires_grad_(True))
        self.device = self.settings['device']

    def my_loss(self, train_features, train_labels):
        """Calculates the loss."""
        return self.loss(self(train_features), train_labels)

    def train(self):
        """Trains the NN according to the settings and saves it to a json file afterwards."""
        if self.settings['nn_type'] == 'MLP_wsymp':
            data, coordinates = self.load_data()
        else:
            data = self.load_data()
        test_data_loaded = False
        self.settings['pos_scaling'] = data.pos_scaling
        self.settings['mom_scaling'] = data.mom_scaling

        if self.settings['optimizer'] == 'adam':
            betas = (self.settings['adam_beta1'], self.settings['adam_beta2'])
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.settings['learning_rate'], betas=betas)
        elif self.settings['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.settings['learning_rate'])
        else:
            raise NotImplementedError

        if self.settings['lr_scheduler'] == 'linear':
            start_factor = 1 - self.settings['initial_epoch']/self.settings['max_epochs']
            total_iters = self.settings['max_epochs'] - self.settings['initial_epoch']
            scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                          start_factor=start_factor,
                                                          end_factor=0,
                                                          total_iters=total_iters,
                                                          verbose=True)
        if self.settings['lr_scheduler'] == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                              0.9,
                                                              verbose=True)
        elif self.settings['lr_scheduler'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                   factor=0.5,
                                                                   patience=100,
                                                                   threshold=0.0001,
                                                                   threshold_mode='rel',
                                                                   cooldown=0,
                                                                   verbose=True)

        if self.settings['loss'] == 'mse':
            if self.settings['nn_type'] == 'MLP_wsymp':
                self.loss = mse_symp_loss(coordinates, self.settings['symp_lambda'])
            elif self.settings['loss_weights']:
                weights = 1/abs(data.train_labels).mean(axis=0)
                self.loss = mse_w_loss(weights)
            else:
                self.loss = torch.nn.MSELoss()
        elif self.settings['loss'] == 'mae':
            if self.settings['nn_type'] == 'MLP_wsymp':
                self.loss = mae_symp_loss(coordinates, self.settings['symp_lambda'])
            elif self.settings['loss_weights']:
                weights = 1/abs(data.train_labels).mean(axis=0)
                self.loss = mae_w_loss(weights)
            else:
                self.loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError

        loss_history = []
        if os.path.isfile(os.path.join(self.settings['path_logs'], 'loss.txt')):
            with open(os.path.join(self.settings['path_logs'], 'loss.txt'), 'r') as file_:
                loss_history = file_.read().split('\n')[:-1]
            loss_history = [elem.split() for elem in loss_history]
            loss_history = [[int(elem[0]), float(elem[1]), float(elem[2])] for elem in loss_history
                            if int(elem[0]) < self.settings['initial_epoch']]

        print('Training...', flush=True)
        for i in range(self.settings['initial_epoch'], self.settings['max_epochs']):
            has_left = True
            while(has_left):
                train_features, train_labels, has_left = data.get_batch(self.settings['batch_size'])
                loss = self.my_loss(train_features, train_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_validation = self.my_loss(data.validation_features, data.validation_labels)
            loss_history.append([i+1, loss.item(), loss_validation.item()])

            if not os.path.isdir(self.settings['path_logs']):
                os.makedirs(self.settings['path_logs'])
            if torch.any(torch.isnan(loss)):
                print('Encountering nan, stop training', flush=True)
                break
            with open(os.path.join(self.settings['path_logs'], 'loss.txt'), 'a') as file_:
                file_.write(f'{i+1:<10} {loss.item():.18e}    {loss_validation.item():.18e}\n')
            print(f'{i+1:<10}Train loss: {loss.item():.18e}    Validation loss: {loss_validation.item():.18e}', flush=True)

            if (i+1) % self.settings['checkpoint_period'] == 0:
                if not os.path.exists(os.path.join(self.settings['model_path'], 'checkpoints')):
                    os.mkdir(os.path.join(self.settings['model_path'], 'checkpoints'))
                self.save_to_json(os.path.join(self.settings['model_path'], 'checkpoints'), f'model{i+1}.json')

            if self.settings['lr_scheduler'] == 'linear' or self.settings['lr_scheduler'] == 'exponential':
                scheduler.step()
            elif self.settings['lr_scheduler'] == 'reduce_on_plateau':
                scheduler.step(loss_validation.item())

        print('Done!', flush=True)

        loss_history = [l for l in loss_history
                        if l[0] in range(self.settings['checkpoint_period'],
                                         self.settings['max_epochs']+1,
                                         self.settings['checkpoint_period'])]
        loss_history = np.array(loss_history)
        best_loss_index = np.argmin(loss_history[:, 1])
        iteration = int(loss_history[best_loss_index, 0])
        loss_train = loss_history[best_loss_index, 1]
        loss_validation = loss_history[best_loss_index, 2]
        print(f'Best model at iteration {iteration}:', flush=True)
        print('Train loss:', loss_train, 'Validation loss:', loss_validation, flush=True)
        self.load_from_json(os.path.join(self.settings['model_path'], 'checkpoints', f'model{iteration}.json'))
        self.save_to_json()
        rmtree(os.path.join(self.settings['model_path'], 'checkpoints'), ignore_errors=True)

    def predict(self, inp):
        """Predicts the labels from the given features.

        Args:
            inp (np.ndarray): Features from which the NN predicts the labels.

        Returns:
            pd.DataFrame: Predicted labels.
        """
        inp = inp.copy()
        if len(inp.shape) == 1:
            inp = inp.reshape((1, inp.shape[0]))
        inp[:,:self.dim] /= self.settings['pos_scaling']
        inp[:,self.dim:] /= self.settings['mom_scaling']
        inp = torch.tensor(inp, dtype=self.dtype, device=self.device)
        predictions = self(inp)
        if self.settings['device'][:3] == 'gpu':
            predictions = pd.DataFrame(predictions.detach().cpu().numpy(), columns=self.settings['label_names'])
        else:
            predictions = pd.DataFrame(predictions.detach().numpy(), columns=self.settings['label_names'])
        predictions[self.get_mom_labels()] *= self.settings['mom_scaling']
        predictions[self.get_pos_labels()] *= self.settings['pos_scaling']
        return predictions

    def predict_path(self, start_pos, end_time, period_q=None):
        """Predicts a whole trajectory from starting positions (and momenta).

        Args:
            start_pos (np.ndarray): Positions (and momenta) at t=0.
              If only postions are given the momenta are set to 0.
            end_time (float): Final time of the trajectory.
            period_q (float): If not None inputs are mapped to [-period_q, period_q].

        Returns:
            pd.DataFrame: Predicted trajectory.
        """
        steps = int(end_time / self.settings['step_size'])
        if len(start_pos.shape) == 1:
            start_pos = start_pos.reshape((1, start_pos.shape[0]))
        coordinates = np.zeros((steps+1, start_pos.shape[0], self.dim*2))
        if start_pos.shape[-1] == self.dim:
            coordinates[0, :, :self.dim] = start_pos
        elif start_pos.shape[-1] == self.dim*2:
            coordinates[0] = start_pos
        else:
            raise ValueError(f'Either give {self.dim} starting positions or {self.dim} positions and {self.dim} momenta')

        for i in range(1, steps+1):
            coordinates[i] = self.predict(coordinates[i-1]).values
            if period_q != None:
                q = [name[0] == 'q' for name in self.settings['feature_names']]
                coordinates[i, :, q] = (coordinates[i, :, q] + period_q) % (2 * period_q) - period_q
        coordinates = np.swapaxes(coordinates,0,1).reshape((-1,coordinates.shape[-1]))
        index = pd.MultiIndex.from_product([range(start_pos.shape[0]), range(steps+1)],
                                           names=["run", "timestep"])
        coordinates = pd.DataFrame(coordinates, columns=self.settings['label_names'], index=index)
        coordinates['time'] = coordinates.index.get_level_values(1) * self.settings['step_size']
        return coordinates

    def calc_mse(self, data, period_q=None, interp='linear'):
        """Calulcates the mean of the MSE at each timestep for trajectories in the dataset.

        Args:
            data (pd.DataFrame): Trajectory data.
            period_q (float): If not None inputs are mapped to [-period_q, period_q].
            interp (str): Type of iterpolation for times in between data points.
              Can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.

        Returns:
            pd.DataFrame: Times and mean of the MSE at each timestep.
        """
        inp = data.xs(0, level='timestep')[self.settings['feature_names']].values

        runs = data.index.get_level_values(0).unique()
        max_time_data = [0.]*len(runs)
        for i, run in enumerate(runs):
            max_time_data[i] = data.loc[run].iloc[-1]['time']
        max_time = max_time_data
        end_time = max(max_time)

        predictions = self.predict_path(inp, end_time, period_q=period_q)
        index = pd.MultiIndex.from_product([runs, predictions.loc[0].index],
                                            names=["run", "timestep"])
        predictions.index = index
        for i, run in enumerate(runs):
            steps = int(max_time[i] / self.settings['step_size'])
            predictions.loc[run] = predictions.loc[(run, slice(0,steps)),:]
        predictions.dropna(how='all', inplace=True)

        if period_q != None:
            q = self.get_pos_features()
            data[q] = (data[q] + period_q) % (2 * period_q) - period_q

        if self.settings['t_in_T']:
            result = pd.DataFrame(np.linspace(0, 1, 101), columns=['time'])
            mean = np.zeros(101)
        else:
            result = pd.DataFrame(predictions.loc[runs[0]]['time'])
            mean = np.zeros(max(predictions.index.get_level_values(1))+1)

        for run in runs:
            f = interp1d(data.loc[run]['time'].values,
                         data.loc[run][self.settings['feature_names']].values,
                         axis=0, kind=interp, fill_value='extrapolate')
            error = f(predictions.loc[run]['time'].values) \
                    - predictions.loc[run][self.settings['feature_names']].values
            error = pd.DataFrame(error, columns=self.settings['feature_names'])

            if period_q != None:
                q = self.get_pos_features()
                error = error.abs()
                error[q] = period_q - (error[q] - period_q).abs()

            if self.settings['t_in_T']:
                f = interp1d(np.linspace(0, 1, len(error.values)), error.values, axis=0, kind=interp)
                error = pd.DataFrame(f(np.linspace(0, 1, 101)), columns=self.settings['feature_names'])

            error = error.values**2
            error = error.mean(axis=-1)

            mean[:len(error)] += error
        result['mse'] = mean/len(runs)

        return result

    def calc_mae(self, data, period_q=None, interp='linear'):
        """Calulcates the mean of the MAE at each timestep for trajectories in the dataset.

        Args:
            data (pd.DataFrame): Trajectory data.
            period_q (float): If not None inputs are mapped to [-period_q, period_q].
            interp (str): Type of iterpolation for times in between data points.
              Can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.

        Returns:
            pd.DataFrame: Times and mean of the MAE at each timestep.
        """
        inp = data.xs(0, level='timestep')[self.settings['feature_names']].values

        runs = data.index.get_level_values(0).unique()
        max_time_data = [0.]*len(runs)
        for i, run in enumerate(runs):
            max_time_data[i] = data.loc[run].iloc[-1]['time']
        max_time = max_time_data
        end_time = max(max_time)

        predictions = self.predict_path(inp, end_time, period_q=period_q)
        index = pd.MultiIndex.from_product([runs, predictions.loc[0].index],
                                            names=["run", "timestep"])
        predictions.index = index
        for i, run in enumerate(runs):
            steps = int(max_time[i] / self.settings['step_size'])
            predictions.loc[run] = predictions.loc[(run, slice(0,steps)),:]
        predictions.dropna(how='all', inplace=True)

        if period_q != None:
            q = self.get_pos_features()
            data[q] = (data[q] + period_q) % (2 * period_q) - period_q

        if self.settings['t_in_T']:
            result = pd.DataFrame(np.linspace(0, 1, 101), columns=['time'])
            mean = np.zeros(101)
        else:
            result = pd.DataFrame(predictions.loc[runs[0]]['time'])
            mean = np.zeros(max(predictions.index.get_level_values(1))+1)

        for run in runs:
            f = interp1d(data.loc[run]['time'].values,
                         data.loc[run][self.settings['feature_names']].values,
                         axis=0, kind=interp, fill_value='extrapolate')
            error = f(predictions.loc[run]['time'].values) \
                    - predictions.loc[run][self.settings['feature_names']].values
            error = pd.DataFrame(error, columns=self.settings['feature_names'])

            if period_q != None:
                q = self.get_pos_features()
                error = error.abs()
                error[q] = period_q - (error[q] - period_q).abs()

            if self.settings['t_in_T']:
                f = interp1d(np.linspace(0, 1, len(error.values)), error.values, axis=0, kind=interp)
                error = pd.DataFrame(f(np.linspace(0, 1, 101)), columns=self.settings['feature_names'])

            error = np.abs(error.values)
            error = error.mean(axis=-1)

            mean[:len(error)] += error
        result['mae'] = mean/len(runs)

        return result

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
