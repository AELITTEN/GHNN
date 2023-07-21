"""Everything needed for the NN's data management."""
from functools import wraps
import math
import numpy as np
import pandas as pd
import torch

__all__ = ['Data_adaptor']

class Data_adaptor:
    """A data adaptor the NNs

    Loads data for NN training from a specified HDF5 store and splits it in training and validation data.
    Manages data type and the device the data is stored on.
    Can spit out minibatches with a specified size.

    Args:
        data_path (str, path-like object, pandas.HDFStore or file-like object): Path to the HDF5 data file.
        max_time (float): Maximum time of the trajectories.
        data_frac (float): Fraction of the data that should be used. Between 0 and 1.
        feature_names (str[]): List of the column names that are used as features.
        label_names (str[]): List of the column names that are used as labels.
        device (str): Device that the data is stored on. 'cpu' or 'gpu'.
        dtype (str): Type of the data. 'float' or 'double'.

    Attributes:
        feature_names (str[]): List of the column names that are used as features.
        label_names (str[]): List of the column names that are used as labels.
        mom_scaling (float): Scaling factor for all momenta.
        pos_scaling (float): Scaling factor for all positions.
        train_features (torch.Tensor): Training data features.
        train_labels (torch.Tensor): Training data labels.
        validation_features (torch.Tensor): Validation data features.
        validation_labels (torch.Tensor): Validation data labels.
        batch_start (int): Starting index for next batch.
        perm (np.ndarray): Current random permutation of the data.
    """

    def __init__(self, data_path, max_time, data_frac, feature_names, label_names, device, dtype):
        # Load the data
        features = pd.read_hdf(data_path, '/features')
        labels = pd.read_hdf(data_path, '/labels')
        val_labels = pd.read_hdf(data_path, '/val_labels')
        val_features = pd.read_hdf(data_path, '/val_features')

        features = features.iloc[:int(features.shape[0]*data_frac)]
        val_features = val_features.iloc[:int(val_features.shape[0]*data_frac)]
        if max_time:
            features = features[features['time'] <= max_time]
            val_features = val_features[val_features['time'] <= max_time]
        labels = labels.loc[features.index]
        val_labels = val_labels.loc[val_features.index]

        fp = features[self.get_mom_features(feature_names)].abs().mean()
        fq = features[self.get_pos_features(feature_names)].abs().mean()
        lp = labels[self.get_mom_labels(label_names)].abs().mean()
        lq = labels[self.get_pos_labels(label_names)].abs().mean()

        mean = (fp.size*fp.mean() + lp.size*lp.mean()) / (fp.size + lp.size)
        mom_order = math.floor(math.log10(mean))
        self.mom_scaling = math.pow(10, mom_order)
        features[self.get_mom_features(feature_names)] /= self.mom_scaling
        labels[self.get_mom_labels(label_names)] /= self.mom_scaling
        val_features[self.get_mom_features(feature_names)] /= self.mom_scaling
        val_labels[self.get_mom_labels(label_names)] /= self.mom_scaling

        mean = (fq.size*fq.mean() + lq.size*lq.mean()) / (fq.size + lq.size)
        pos_order = math.floor(math.log10(mean))
        self.pos_scaling = math.pow(10, pos_order)
        features[self.get_pos_features(feature_names)] /= self.pos_scaling
        labels[self.get_pos_labels(label_names)] /= self.pos_scaling
        val_features[self.get_pos_features(feature_names)] /= self.pos_scaling
        val_labels[self.get_pos_labels(label_names)] /= self.pos_scaling

        if 'time' in feature_names:
            features['time'] /= self.pos_scaling / self.mom_scaling
            val_features['time'] /= self.pos_scaling / self.mom_scaling

        self.train_features = features[feature_names].values
        self.train_labels = labels[label_names].values
        self.validation_features = val_features[feature_names].values
        self.validation_labels = val_labels[label_names].values

        # Set the device
        if device == 'cpu':
            for d in ['train_features', 'train_labels', 'validation_features', 'validation_labels']:
                if isinstance(getattr(self, d), np.ndarray):
                    setattr(self, d, torch.DoubleTensor(getattr(self, d)))
                elif isinstance(getattr(self, d), torch.Tensor):
                    setattr(self, d, getattr(self, d).cpu())
        elif device == 'gpu':
            for d in ['train_features', 'train_labels', 'validation_features', 'validation_labels']:
                if isinstance(getattr(self, d), np.ndarray):
                    setattr(self, d, torch.cuda.DoubleTensor(getattr(self, d)))
                elif isinstance(getattr(self, d), torch.Tensor):
                    setattr(self, d, getattr(self, d).cuda())
        elif device[:3] == 'gpu':
            index = int(device[3:])
            for d in ['train_features', 'train_labels', 'validation_features', 'validation_labels']:
                if isinstance(getattr(self, d), np.ndarray):
                    setattr(self, d, torch.DoubleTensor(getattr(self, d)).cuda(index))
                elif isinstance(getattr(self, d), torch.Tensor):
                    setattr(self, d, getattr(self, d).cuda(index))
        else:
            raise ValueError

        # Adjust the data type
        if dtype == 'float':
            for d in ['train_features', 'train_labels', 'validation_features', 'validation_labels']:
                setattr(self, d, getattr(self, d).float())
        elif dtype == 'double':
            for d in ['train_features', 'train_labels', 'validation_features', 'validation_labels']:
                setattr(self, d, getattr(self, d).double())
        else:
            raise ValueError

        self.batch_start = 0
        self.perm = np.arange(self.train_features.size(0))

    def get_pos_features(self, feature_names):
        """Returns a list of the position feature names."""
        return [feat for feat in feature_names if feat[0] == 'q']

    def get_pos_labels(self, label_names):
        """Returns a list of the position label names."""
        return [lab for lab in label_names if lab[0] == 'q']

    def get_mom_features(self, feature_names):
        """Returns a list of the momentum feature names."""
        return [feat for feat in feature_names if feat[0] == 'p']

    def get_mom_labels(self, label_names):
        """Returns a list of the momentum label names."""
        return [lab for lab in label_names if lab[0] == 'p']

    def get_batch(self, batch_size, shuffle=True):
        """Returns one random minibatch of the training data of size batch_size.

        Iterates through the data. If it is the first batch, the data is shuffled.
        Throws away last bit of data that does not fill an entire batch.
        So, please do not change batch_size within one epoch.

        Args:
            batch_size (int): Size of the minibatch.
            shuffle (bool): Whether to shuffle the data every epoch or not.

        Returns:
            3-element tuple:
            - (*torch.Tensor*): Features for this batch.
            - (*torch.Tensor*): Labels for this batch.
            - (*bool*): Whether there are batches left for this epoch.
        """
        if batch_size is None:
            return self.train_features, self.train_labels, False
        else:
            has_left = True
            if self.batch_start == 0 and shuffle:
                np.random.shuffle(self.perm)
            mask = self.perm[self.batch_start:self.batch_start+batch_size]
            self.batch_start += batch_size
            if self.batch_start + batch_size >= self.train_features.size(0):
                has_left = False
                self.batch_start = 0
            return self.train_features[mask], self.train_labels[mask], has_left
