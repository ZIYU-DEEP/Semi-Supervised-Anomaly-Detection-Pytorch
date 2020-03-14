"""
Title: forecast_dataset.py
Description: The dataset classes for the forecast model.
Author: Leksai Ye, University of Chicago
"""

import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


# --------------------------------------------
# 0. Helper Functions
# --------------------------------------------
def window_data(array, in_size, out_size, overlap=False, n_features=128):
    """
    Input:
        array (np.array): shape is (n_instances, n_features)
        in_size (int): the size (no. of instances) for a window's prefix
        out_size (int): the size for a window's suffix
        n_features (int): the no. of features
    Returns:
        array_in (np.array): shape is (n_instances, in_size, n_features)
        array_out (np.array): shape is (n_instances, out_size, n_features)
    """
    if len(array.shape) > 2:
        array = array.reshape(-1, n_features)

    len_ = len(array)
    win_size = in_size + out_size

    if overlap:
        start_index = [i for i in range(0, len_, out_size)]
    else:
        start_index = [i for i in range(0, len_, win_size)]
    if start_index[-1] + win_size > len_: start_index.pop()

    index_in = [list(range(i, i + in_size)) for i in start_index]
    index_out = [list(range(i + in_size, i + in_size + out_size)) for i in start_index]

    array_in = array[index_in, :]
    array_out = array[index_out, :]

    return array_in, array_out


# --------------------------------------------
# 1.1. (a) The Dataset Object for Training
# --------------------------------------------
class ForecastDataset(Dataset):
    def __init__(self,
                 root: str,
                 normal_filename: str,
                 abnormal_filename: str,
                 random_state=42,
                 in_size=100,
                 out_size=25,
                 n_features=128,
                 train_portion=0.8,
                 train=False):
        super(Dataset, self).__init__()

        # Pre-processing
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)

        self.root = Path(root)
        self.normal_path = self.root / normal_filename
        self.abnormal_path = self.root / abnormal_filename
        self.train = train
        np.random.seed(random_state)

        # Loading Data
        print('Loading abnormal data...')
        X_posi = np.load(self.abnormal_path)
        print('Abnormal data: I am loaded!')

        print('Loading normal data...')
        X_nega = np.load(self.normal_path)
        print('Normal data: I am loaded!')

        # Train-test split
        # Note that we are not doing random shuffling here
        # As our data is time-seires alike
        X_posi_train = X_posi[:int(len(X_posi) * train_portion)]
        X_posi_test = X_posi[int(len(X_posi) * train_portion):]

        X_nega_train = X_nega[:int(len(X_nega) * train_portion)]
        X_nega_test = X_nega[int(len(X_nega) * train_portion):]

        if self.train:
            # Window the data for training
            print('Get windows of the data!')
            X_posi_train_in, X_posi_train_out = window_data(X_posi_train, in_size,
                                                            out_size, False, n_features)
            X_nega_train_in, X_nega_train_out = window_data(X_nega_train, in_size,
                                                            out_size, False, n_features)

            # This is because X_in and X_out should have the same label
            y_posi_train = np.ones(X_posi_train_in.shape[0])
            y_nega_train = np.zeros(X_nega_train_in.shape[0])

            # Concatenating data
            print('Concatenating data!')
            self.X_in = torch.tensor(np.concatenate((X_posi_train_in, X_nega_train_in)),
                                     dtype=torch.float64)
            self.X_out = torch.tensor(np.concatenate((X_posi_train_out, X_nega_train_out)),
                                      dtype=torch.float64)
            self.y = torch.tensor(np.concatenate((y_posi_train, y_nega_train)),
                                  dtype=torch.int64)

        if not self.train:
            # Window the data for testing
            print('Get windows of the data!')
            X_posi_test_in, X_posi_test_out = window_data(X_posi_test, in_size,
                                                          out_size, False, n_features)
            X_nega_test_in, X_nega_test_out = window_data(X_nega_test, in_size,
                                                            out_size, False, n_features)

            y_posi_test = np.ones(X_posi_test_in.shape[0])
            y_nega_test = np.zeros(X_nega_test_in.shape[0])

            # Concatenating data
            print('Concatenating data!')
            self.X_in = torch.tensor(np.concatenate((X_posi_test_in, X_nega_test_in)),
                                     dtype=torch.float64)
            self.X_out = torch.tensor(np.concatenate((X_posi_test_out, X_nega_test_out)),
                                      dtype=torch.float64)
            self.y = torch.tensor(np.concatenate((y_posi_test, y_nega_test)),
                                  dtype=torch.float64)

    def __getitem__(self, index):
        X_in, X_out, y = self.X_in[index], self.X_out[index], int(self.y[index])
        return X_in, X_out, y, index

    def __len__(self):
        return len(self.X_in)


# --------------------------------------------
# 1.1. (b) The Dataset Object for Training (Unsupervised Version)
# --------------------------------------------
class ForecastDataset_(Dataset):
    def __init__(self,
                 root: str,
                 normal_filename: str,
                 abnormal_filename: str = '_',                 
                 random_state=42,
                 in_size=100,
                 out_size=25,
                 n_features=128,
                 train_portion=0.8,
                 train=False):
        super(Dataset, self).__init__()

        # Pre-processing
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)

        self.root = Path(root)
        self.normal_path = self.root / normal_filename
        self.train = train
        np.random.seed(random_state)

        # Loading Data
        print('Loading normal data...')
        X_nega = np.load(self.normal_path)
        print('Normal data: I am loaded!')

        # Train-test split
        # Note that we are not doing random shuffling here
        # As our data is time-seires alike
        X_nega_train = X_nega[:int(len(X_nega) * train_portion)]
        X_nega_test = X_nega[int(len(X_nega) * train_portion):]

        if self.train:
            # Window the data for training
            print('Get windows of the data!')
            X_nega_train_in, X_nega_train_out = window_data(X_nega_train, in_size,
                                                            out_size, False, n_features)

            # This is because X_in and X_out should have the same label
            y_nega_train = np.zeros(X_nega_train_in.shape[0])

            # Concatenating data
            print('Concatenating data!')
            self.X_in = torch.tensor(X_nega_train_in, dtype=torch.float64)
            self.X_out = torch.tensor(X_nega_train_in, dtype=torch.float64)
            self.y = torch.tensor(y_nega_train, dtype=torch.int64)

        if not self.train:
            # Window the data for testing
            print('Get windows of the data!')
            X_nega_test_in, X_nega_test_out = window_data(X_nega_test, in_size,
                                                          out_size, False, n_features)
            y_nega_test = np.zeros(X_nega_test_in.shape[0])

            # Concatenating data
            print('Concatenating data!')
            self.X_in = torch.tensor(X_nega_test_in, dtype=torch.float64)
            self.X_out = torch.tensor(X_nega_test_out, dtype=torch.float64)
            self.y = torch.tensor(y_nega_test, dtype=torch.float64)

    def __getitem__(self, index):
        X_in, X_out, y = self.X_in[index], self.X_out[index], int(self.y[index])
        return X_in, X_out, y, index

    def __len__(self):
        return len(self.X_in)


# --------------------------------------------
# 1.1. (c) The Dataset Object for Evaluation
# --------------------------------------------
class ForecastDatasetEval(Dataset):
    def __init__(self,
                 root: str,
                 abnormal_filename: str,
                 random_state=42,
                 in_size=100,
                 out_size=25,
                 n_features=128):
        super(Dataset, self).__init__()

        # Preprocessing
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)

        self.root = Path(root)
        self.abnormal_path = self.root / abnormal_filename
        np.random.seed(random_state)

        # Loading Data
        print('Loading abnormal data...')
        X_posi = np.load(self.abnormal_path)
        print('Abnormal data: I am loaded!')

        # Window the data for training
        print('Get windows of the data!')
        X_posi_in, X_posi_out = window_data(X_posi, in_size,
                                            out_size, False, n_features)
        y_posi = np.ones(X_posi_in.shape[0])

        # Concatenating data
        print('Getting in and out data!')
        self.X_in = torch.tensor(X_posi_in, dtype=torch.float64)
        self.X_out = torch.tensor(X_posi_out, dtype=torch.float64)
        self.y = torch.tensor(y_posi, dtype=torch.int64)

    def __getitem__(self, index):
        X_in, X_out, y = self.X_in[index], self.X_out[index], int(self.y[index])
        return X_in, X_out, y, index

    def __len__(self):
        return len(self.X_in)
