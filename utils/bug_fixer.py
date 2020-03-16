#############################################
# 0. Preparation
#############################################
import os
import sys
import json
import glob
import time
import torch
import click
import random
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

normal_file = str(sys.argv[1])  # 'ryerson_train_big_normal'
abnormal_file = str(sys.argv[2])  # 'ryerson_ab_train_sigOver_3199_abnormal'
df_name = str(sys.argv[3])  # 'ryerson_train_big_sigOver_3199'
test_list_filename = str(sys.argv[4])  # 'ryerson_test_list.npy'
device_no = str(sys.argv[5])  # 0
eta_str = int(sys.argv[6]) # 50
n_epochs = int(sys.argv[7]) # 250

root = '/net/adv_spectrum/array_data'
net_name = 'lstm_100'
eta = eta_str * 0.01
lr_milestones = (80, 120)
in_size = 100
out_size = 25
n_features = 128
random_state = 42
train_portion = 0.8
device = 'cuda:{}'.format(device_no)

xp_path = './{}'.format(df_name)
txt_result_file = './full_results.txt'
model_path = xp_path + '/model_{}_{}_{}_{}.tar'.format(net_name, df_name, n_epochs, eta_str)
results_path = xp_path + '/results_{}_{}_{}_{}.json'.format(net_name, df_name, n_epochs, eta_str)
cut_path = xp_path + '/cut_{}_{}_{}_{}.npy'.format(net_name, df_name, n_epochs, eta_str)
normal_filename = normal_file + '.npy'
abnormal_filename = abnormal_file + '.npy'
test_list = np.load(test_list_filename)

if not os.path.exists(xp_path):
    os.makedirs(xp_path)

#############################################
# 0. Utils Functions
#############################################
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

#############################################
# 1. Functions for Custom Dataset
#############################################
#--------------------------------------------
# 1.1. (a) The Dataset Object for Training
#--------------------------------------------
class CustomDataset(Dataset):
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

        # Preprocessing
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
                                     dtype=torch.float32)
            self.X_out = torch.tensor(np.concatenate((X_posi_train_out, X_nega_train_out)),
                                      dtype=torch.float32)
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
                                     dtype=torch.float32)
            self.X_out = torch.tensor(np.concatenate((X_posi_test_out, X_nega_test_out)),
                                      dtype=torch.float32)
            self.y = torch.tensor(np.concatenate((y_posi_test, y_nega_test)),
                                  dtype=torch.float32)

    def __getitem__(self, index):
        X_in, X_out, y = self.X_in[index], self.X_out[index], int(self.y[index])
        return X_in, X_out, y, index

    def __len__(self):
        return len(self.X_in)


#--------------------------------------------
# 1.1. (b) The Dataset Object for Evaluation
#--------------------------------------------
class CustomDataset_eval(Dataset):
    def __init__(self,
                 root: str,
                 abnormal_filename: str,
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
        X_posi_in, X_posi_out = window_data(X_posi, in_size, out_size, False, n_features)
        y_posi = np.ones(X_posi_in.shape[0])

        # Concatenating data
        print('Getting in and out data!')
        self.X_in = torch.tensor(X_posi_in, dtype=torch.float32)
        self.X_out = torch.tensor(X_posi_out, dtype=torch.float32)
        self.y = torch.tensor(y_posi, dtype=torch.int64)


    def __getitem__(self, index):
        X_in, X_out, y = self.X_in[index], self.X_out[index], int(self.y[index])
        return X_in, X_out, y, index

    def __len__(self):
        return len(self.X_in)


#--------------------------------------------
# 1.2. (a) The Loader Object for Training
#--------------------------------------------
from torch.utils.data import DataLoader

class CustomLoader():
    def __init__(self,
                 root: str,
                 normal_filename: str,
                 abnormal_filename: str,
                 random_state=42,
                 in_size=100,
                 out_size=25,
                 n_features=128,
                 train_portion=0.8):

        print('Hi! I am setting train_set for you.')
        self.train_set = CustomDataset(root,
                                       normal_filename,
                                       abnormal_filename,
                                       random_state,
                                       in_size,
                                       out_size,
                                       n_features,
                                       train_portion,
                                       train=True,)

        print('\nHi! I am setting test_set for you.')
        self.test_set = CustomDataset(root,
                                      normal_filename,
                                      abnormal_filename,
                                      random_state,
                                      in_size,
                                      out_size,
                                      n_features,
                                      train_portion,
                                      train=False)
    def loaders(self,
                batch_size: int,
                shuffle_train=True,
                shuffle_test=False,
                num_workers: int = 0) -> (DataLoader, DataLoader):

        train_loader = DataLoader(dataset=self.train_set,
                                  batch_size=batch_size,
                                  shuffle=shuffle_train,
                                  num_workers=num_workers,
                                  drop_last=True)
        test_loader = DataLoader(dataset=self.test_set,
                                 batch_size=batch_size,
                                 shuffle=shuffle_test,
                                 num_workers=num_workers,
                                 drop_last=False)
        return train_loader, test_loader

    def __repr__(self):
        return self.__class__.__name__


#--------------------------------------------
# 1.2. (b) The Loader Object for Testing
#--------------------------------------------
from torch.utils.data import DataLoader

class CustomLoader_eval():
    def __init__(self,
                 root: str,
                 abnormal_filename: str,
                 in_size=100,
                 out_size=25,
                 n_features=128):

        print('Hi! I am setting train_set for you.')
        self.all_set = CustomDataset_eval(root,
                                          abnormal_filename,
                                          in_size,
                                          out_size,
                                          n_features)


    def loaders(self,
                batch_size: int,
                shuffle_all=False,
                num_workers: int = 0) -> (DataLoader, DataLoader):

        all_loader = DataLoader(dataset=self.all_set,
                                batch_size=batch_size,
                                shuffle=shuffle_all,
                                num_workers=num_workers,
                                drop_last=True)
        return all_loader

    def __repr__(self):
        return self.__class__.__name__


#############################################
# 2. Classes and Function for Networks
#############################################
#--------------------------------------------
# 2.1. Base Network
#--------------------------------------------
import logging
import torch.nn as nn
import numpy as np


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the code layer or last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


#--------------------------------------------
# 2.2. LSTM Network
#--------------------------------------------
class LSTM_Net_100(BaseNet):

    def __init__(self):
        super().__init__()

        self.lstm = nn.GRU(128, 64, num_layers=1, batch_first=True)
        self.BN = nn.BatchNorm1d(100)
        self.fc = nn.Linear(64 * 100, 128 * 25)
        self.act = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.BN(x)
        x = x.reshape(x.size(0), 64 * 100).contiguous()
        x = self.fc(x)
        x = x.reshape(-1, 25, 128).contiguous()
        x = self.act(x)
        return x

class LSTM_Net_100_stacked(BaseNet):

    def __init__(self):
        super().__init__()

        self.lstm1 = nn.GRU(128, 64, num_layers=1, batch_first=True)
        self.BN1 = nn.BatchNorm1d(100)
        self.lstm2 = nn.GRU(64, 64, num_layers=1, batch_first=True)
        self.BN2 = nn.BatchNorm1d(100)
        self.lstm3 = nn.GRU(64, 64, num_layers=1, batch_first=True)
        self.BN3 = nn.BatchNorm1d(100)
        self.fc = nn.Linear(64 * 100, 128 * 25)
        self.act = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.BN1(x)
        x, _ = self.lstm2(x)
        x = self.BN2(x)
        x, _ = self.lstm3(x)
        x = self.BN3(x)
        x = x.reshape(x.size(0), 64 * 100).contiguous()
        x = self.fc(x)
        x = x.reshape(-1, 25, 128).contiguous()
        x = self.act(x)
        return x

#--------------------------------------------
# 2.3. Function to Build Network
#--------------------------------------------
def build_network(net_name='lstm'):
    """Builds the neural network."""
    net_name=net_name.strip()
    net = None

    if net_name == "lstm_100": return LSTM_Net_100()
    if net_name == "lstm_100_stacked": return LSTM_Net_100_stacked()


#############################################
# 3. Classes and Function for Trainers
#############################################
#--------------------------------------------
# 3.1. Base Trainer
#--------------------------------------------
class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset, net):
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, dataset, net):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass


class BaseEvaluater(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader


    @abstractmethod
    def test(self, dataset, net):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass


#--------------------------------------------
# 3.2. (a) Trainer
#--------------------------------------------
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import time
import torch
import torch.optim as optim
import numpy as np


class Trainer(BaseTrainer):

    def __init__(self,
                 eta: float = 1,
                 optimizer_name: str = 'adam',
                 lr: float = 0.01,
                 n_epochs: int = 60,
                 lr_milestones: tuple = (50, 100, 150, 200),
                 batch_size: int = 32,
                 weight_decay: float = 1e-6,
                 device: str = 'cuda:1',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.eta = eta

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset, net):

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size,
                                          num_workers=self.n_jobs_dataloader)

        # Set device for network
        print('Loading net...')
        net = net.to(self.device)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.lr_milestones,
                                                   gamma=0.1)

        # Training
        print('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' %
                            float(scheduler.get_lr()[0]))

            epoch_loss, n_batches = 0.0, 0
            epoch_start_time = time.time()
            for data in train_loader:
                X_in, X_out, y, _ = data
                X_in, X_out, y = X_in.to(self.device), X_out.to(self.device), y.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                X_pred = net(X_in)
                dist = criterion(X_pred, X_out)
                dist_mean = torch.mean(dist, axis=[1, 2])
                losses = torch.where(y == 0,
                                     dist_mean,
                                     self.eta * ((dist_mean) ** (-1)))

                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | '
                        f'Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        print('Training Time: {:.3f}s'.format(self.train_time))
        print('Finished training.')

        return net

    def test(self, dataset, net):

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size,
                                         num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Testing
        print('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                X_in, X_out, y, idx = data
                X_in, X_out = X_in.to(self.device), X_out.to(self.device)
                y, idx = y.to(self.device), idx.to(self.device)

                X_pred = net(X_in)
                dist = criterion(X_pred, X_out)

                dist_mean = torch.mean(dist, axis=[1, 2])
                losses = torch.where(y == 0,
                                     dist_mean,
                                     self.eta * ((dist_mean) ** (-1)))

                loss = torch.mean(losses)
                scores = torch.mean(dist, axis=[1, 2])

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            y.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        print('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing.')


#--------------------------------------------
# 3.2. (b) Evaluater
#--------------------------------------------
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import time
import torch
import torch.optim as optim
import numpy as np


class Evaluater(BaseEvaluater):

    def __init__(self,
                 eta: float = 1,
                 optimizer_name: str = 'adam',
                 lr: float = 0.001,
                 n_epochs: int = 60,
                 lr_milestones: tuple = (80, 120),
                 batch_size: int = 32,
                 weight_decay: float = 1e-6,
                 device: str = 'cuda:1',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.eta = eta

        # Results
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def test(self, dataset, net):

        # Get test data loader
        all_loader = dataset.loaders(batch_size=self.batch_size,
                                     num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Testing
        print('Starting evaluating...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in all_loader:
                X_in, X_out, y, idx = data
                X_in, X_out = X_in.to(self.device), X_out.to(self.device)
                y, idx = y.to(self.device), idx.to(self.device)

                X_pred = net(X_in)
                dist = criterion(X_pred, X_out)

                dist_mean = torch.mean(dist, axis=[1, 2])
                losses = torch.where(y == 0,
                                     dist_mean,
                                     self.eta * ((dist_mean) ** (-1)))

                loss = torch.mean(losses)
                scores = torch.mean(dist, axis=[1, 2])

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            y.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        # Log results
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing.')


#--------------------------------------------
# 3.3. (a) Model Object for training
#--------------------------------------------
class LSTM_Forecast(object):
    """A class for the Deep SAD method.
    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.net_name = None
        self.net = None
        self.trainer = None
        self.optimizer_name = None

        self.results = {'train_time': None, 'test_auc': None,
                        'test_time': None,'test_scores': None}

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self,
              dataset,
              eta: float = 1,
              optimizer_name: str = 'adam',
              lr: float = 0.001,
              n_epochs: int = 60,
              lr_milestones: tuple = (100, 160, 220),
              batch_size: int = 32,
              weight_decay: float = 1e-6,
              device: str = 'cuda:1',
              n_jobs_dataloader: int = 0):
        print('Learning rate: {}'.format(lr))
        self.optimizer_name = optimizer_name
        self.trainer = Trainer(eta, optimizer_name, lr, n_epochs, lr_milestones, batch_size,
                               weight_decay, device, n_jobs_dataloader)
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time


    def test(self, dataset, device: str = 'cuda:1', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        self.trainer.test(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def save_model(self, export_model, save_ae=True):
        """Save Deep SAD model to export_model."""

        net_dict = self.net.state_dict()

        torch.save({'net_dict': net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)
        self.net.load_state_dict(model_dict['net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)


#--------------------------------------------
# 3.3. (b) Model Object for Evaluating
#--------------------------------------------
class LSTM_Forecast_eval(object):
    """A class for the Deep SAD method.
    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.net = None
        self.evaluater= None
        self.optimizer_name = None
        self.results = {'test_time': None,'test_scores': None}

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def load_model(self, model_path, map_location='cpu'):

        model_dict = torch.load(model_path, map_location=map_location)
        self.net.load_state_dict(model_dict['net_dict'])

    def test(self,
             dataset,
             eta: float = 1,
             optimizer_name: str = 'adam',
             lr: float = 0.001,
             n_epochs: int = 60,
             lr_milestones: tuple = (100, 160, 220),
             batch_size: int = 32,
             weight_decay: float = 1e-6,
             device: str = 'cuda:1',
             n_jobs_dataloader: int = 0):

        if self.evaluater is None:
            self.evaluater = Evaluater(eta=eta, weight_decay=weight_decay, device=device)

        self.evaluater.test(dataset, self.net)

        # Get results
        self.results['test_time'] = self.evaluater.test_time
        self.results['test_scores'] = self.evaluater.test_scores

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)


#############################################
# 4. Main
#############################################
#--------------------------------------------
# 4.1. Main Training
#--------------------------------------------
dataset = CustomLoader(root,
                       normal_filename,
                       abnormal_filename,
                       random_state,
                       in_size,
                       out_size,
                       n_features,
                       train_portion)

model = LSTM_Forecast()
model.set_network(net_name)
model.train(dataset, n_epochs=n_epochs, lr_milestones=lr_milestones, eta=eta, device=device)
model.test(dataset)
model.save_results(export_json=results_path)
model.save_model(export_model= model_path)

#--------------------------------------------
# 4.2. Main Testing
#--------------------------------------------
train_auc = model.results['test_auc']
indices, labels, scores = zip(*model.results['test_scores'])
indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
result_df = pd.DataFrame()
result_df['indices'] = indices
result_df['labels'] = labels
result_df['scores'] = scores
result_df_path = '{}/result_df_{}_{}_{}_{}.pkl'.format(xp_path, net_name, df_name, n_epochs, eta_str)
result_df.to_pickle(result_df_path)

# Write the file for detection rate
result_df.drop('indices', inplace=True, axis=1)
df_normal = result_df[result_df.labels == 0]
df_abnormal = result_df[result_df.labels == 1]
cut = df_normal.scores.quantile(0.95)
np.save(cut_path, cut)
y = [1 if e > cut else 0 for e in df_abnormal['scores'].values]
f = open(txt_result_file, 'a')
f.write('=====================\n')
f.write('[DataFrame Name] {}\n'.format(result_df_path))
f.write('[Eta] {}\n'.format(eta))
f.write('[Normal to Abnormal Ratio] 1:{}\n'.format(
    len(df_abnormal) / len(df_normal)))
f.write('[False Positive Rate] 0.05\n')
f.write('[Train AUC] {}\n'.format(train_auc))
f.write('---------------------\n')
f.write('[Recall for {}] {}\n'.format('TEST', sum(y) / len(y)))

#--------------------------------------------
# 4.2. Main Evaluating
#--------------------------------------------
# cut = np.load(cut_path)[0]
for test_abnormal_filename in test_list:
    dataset_eval = CustomLoader_eval(root,
                                     test_abnormal_filename,
                                     in_size,
                                     out_size,
                                     n_features)
    model_eval = LSTM_Forecast_eval(eta=eta)
    model_eval.set_network(net_name)
    model_eval.load_model(model_path=model_path, map_location=device)
    model_eval.test(dataset_eval, device=device, eta=eta)
    indices, labels, scores = zip(*model_eval.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    result_df = pd.DataFrame()
    result_df['indices'] = indices
    result_df['labels'] = labels
    result_df['scores'] = scores
    y = [1 if e > cut else 0 for e in scores]
    print('Detection result for the file: {}'.format(test_abnormal_filename))
    print(sum(y) / len(y))
    f.write('---------------------\n')
    f.write('[Recall for {}] {}\n'.format(test_abnormal_filename, sum(y) / len(y)))

f.write('=====================\n\n')
f.close()
