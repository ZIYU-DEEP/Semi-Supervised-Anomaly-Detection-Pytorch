"""
Title: forecast_dataset.py
Description: The dataset classes for the forecast model.
Author: Lek'Sai Ye, University of Chicago
"""

import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


# --------------------------------------------
# Dataset for Semi-Supervised Forecast Model (root, abnormal_folder, train)
# --------------------------------------------
class ForecastDataset(Dataset):
    def __init__(self,
                 root: str='/net/adv_spectrum/torch_data',
                 normal_folder: str='downtown',
                 abnormal_folder: str='downtown_sigOver_10ms',
                 train: bool=True):
        super(Dataset, self).__init__()

        print('Hey - I am starting to load your data!')
        data_source = Path(root) / normal_folder

        if train:
            X_nega_in = np.load(data_source / 'normal' / 'X_train_in.npy')
            X_nega_out = np.load(data_source / 'normal' / 'X_train_out.npy')

            X_posi_in = np.load(data_source / 'abnormal' / abnormal_folder / 'X_train_in.npy')
            X_posi_out = np.load(data_source / 'abnormal' / abnormal_folder / 'X_train_out.npy')
        else:
            X_nega_in = np.load(data_source / 'normal' / 'X_test_in.npy')
            X_nega_out = np.load(data_source / 'normal' / 'X_test_out.npy')

            X_posi_in = np.load(data_source / 'abnormal' / abnormal_folder / 'X_test_in.npy')
            X_posi_out = np.load(data_source / 'abnormal' / abnormal_folder / 'X_test_out.npy')

        y_nega = np.zeros(X_nega_in.shape[0])
        y_posi = np.ones(X_posi_in.shape[0])

        print('Concatenating data!')
        # Note that we do not do shuffling here
        # Rather, we will shuffle them at the DataLoader
        self.X_in = torch.tensor(np.concatenate((X_nega_in, X_posi_in)), dtype=torch.float32)
        self.X_out = torch.tensor(np.concatenate((X_nega_out, X_posi_out)), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate((y_nega, y_posi)), dtype=torch.int32)
        print('Data loaded!')

    def __getitem__(self, index):
        X_in, X_out, y = self.X_in[index], self.X_out[index], int(self.y[index])
        return X_in, X_out, y, index

    def __len__(self):
        return len(self.X_in)


# --------------------------------------------
# Dataset for Unsupervised Forecast Model (root, train)
# --------------------------------------------
class ForecastDatasetUnsupervised(Dataset):
    def __init__(self,
                 root: str='/net/adv_spectrum/torch_data',
                 normal_folder: str='downtown',
                 train: bool=True):
        super(Dataset, self).__init__()

        print('Hey - I am starting to load your data!')
        data_source = Path(root) / normal_folder

        if train:
            X_in = np.load(data_source / 'normal' / 'X_train_in.npy')
            X_out = np.load(data_source / 'normal' / 'X_train_out.npy')
        else:
            X_in = np.load(data_source / 'normal' / 'X_test_in.npy')
            X_out = np.load(data_source / 'normal' / 'X_test_out.npy')

        y = np.zeros(X_in.shape[0])

        self.X_in = torch.tensor(X_in, dtype=torch.float32)
        self.X_out = torch.tensor(X_out, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int32)
        print('Data loaded!')

    def __getitem__(self, index):
        X_in, X_out, y = self.X_in[index], self.X_out[index], int(self.y[index])
        return X_in, X_out, y, index

    def __len__(self):
        return len(self.X_in)


# --------------------------------------------
# Dataset for Evaluation of Forecast Model (folder)
# --------------------------------------------
class ForecastDatasetEval(Dataset):
    def __init__(self,
                 root: str='/net/adv_spectrum/torch_data/downtown/abnormal/downtown_LOS-5M-USRP1/file_0'):
        super(Dataset, self).__init__()
        # Be aware that this 'root' is different from
        # the previous root in our setting

        X_in = np.load(Path(root) / 'X_in.npy')
        X_out = np.load(Path(root) / 'X_out.npy')
        y = np.ones(X_in.shape[0])

        self.X_in = torch.tensor(X_in, dtype=torch.float32)
        self.X_out = torch.tensor(X_out, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int32)

    def __getitem__(self, index):
        X_in, X_out, y = self.X_in[index], self.X_out[index], int(self.y[index])
        return X_in, X_out, y, index

    def __len__(self):
        return len(self.X_in)
