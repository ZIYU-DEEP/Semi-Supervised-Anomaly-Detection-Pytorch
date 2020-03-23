"""
Title: rec_dataset.py
Description: The dataset classes for the reconstruction model.
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
class RecDataset(Dataset):
    def __init__(self,
                 root: str='/net/adv_spectrum/torch_data_deepsad/100',
                 normal_folder: str='ryerson_train',
                 abnormal_folder: str='ryerson_ab_train_sigOver_10ms',
                 train: bool=True):
        super(Dataset, self).__init__()

        print('Hey - I am starting to load your data!')
        data_source = Path(root) / normal_folder

        if train:
            X_nega = np.load(data_source / 'normal' / 'X_train.npy')
            X_posi = np.load(data_source / 'abnormal' / abnormal_folder / 'X_train.npy')
        else:
            X_nega = np.load(data_source / 'normal' / 'X_test.npy')
            X_posi = np.load(data_source / 'abnormal' / abnormal_folder / 'X_test.npy')

        y_nega = np.zeros(X_nega.shape[0])
        y_posi = np.ones(X_posi.shape[0])

        # This line is optional, just to limit the data for anomaly
        if len(X_posi) > len(X_nega):
            X_posi = X_posi[: len(X_nega)]

        print('Concatenating data!')
        # Note that we do not do shuffling here
        # Rather, we will shuffle them at the DataLoader
        self.X = torch.tensor(np.concatenate((X_nega, X_posi)), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate((y_nega, y_posi)), dtype=torch.int32)
        print('Data loaded!')

    def __getitem__(self, index):
        X, y = self.X[index], int(self.y[index])
        return X, y, index

    def __len__(self):
        return len(self.X)


# --------------------------------------------
# Dataset for Unsupervised Forecast Model (root, train)
# --------------------------------------------
class RecDatasetUnsupervised(Dataset):
    def __init__(self,
                 root: str='/net/adv_spectrum/torch_data_deepsad/100',
                 normal_folder: str='ryerson_train',
                 train: bool=True):
        super(Dataset, self).__init__()

        print('Hey - I am starting to load your data!')
        data_source = Path(root) / normal_folder

        if train:
            X = np.load(data_source / 'normal' / 'X_train.npy')
        else:
            X = np.load(data_source / 'normal' / 'X_test.npy')

        y = np.zeros(X.shape[0])

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int32)
        print('Data loaded!')

    def __getitem__(self, index):
        X, y = self.X[index], int(self.y[index])
        return X, y, index

    def __len__(self):
        return len(self.X)


# --------------------------------------------
# Dataset for Evaluation of Forecast Model (folder)
# --------------------------------------------
class RecDatasetEval(Dataset):
    def __init__(self,
                 root: str='/net/adv_spectrum/torch_data_deepsad/100/ryerson_train/abnormal/ryerson_ab_train_LOS-5M-USRP1/file_0'):
        super(Dataset, self).__init__()
        # Be aware that this 'root' is different from
        # the previous root in our setting

        X = np.load(Path(root) / 'X.npy')
        y = np.ones(X.shape[0])

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int32)

    def __getitem__(self, index):
        X, y = self.X[index], int(self.y[index])
        return X, y, index

    def __len__(self):
        return len(self.X)
