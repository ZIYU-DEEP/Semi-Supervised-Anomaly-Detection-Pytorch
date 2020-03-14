"""
Title: forecast_optimizer_unsupervised.py
Description: The unsupervised optimizer.
Author: Leksai Ye
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/optim
"""

from base_optimizer import BaseTrainer, BaseEvaluater
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import time


# --------------------------------------------
# 3.2. (1) Trainer
# --------------------------------------------
class ForecastTrainer_(BaseTrainer):
    def __init__(self,
                 eta: float= 1,
                 optimizer_name: str = 'adam',
                 lr: float = 0.01,
                 n_epochs: int = 60,
                 lr_milestones: tuple = (50, 100, 150, 200),
                 batch_size: int = 32,
                 weight_decay: float = 1e-6,
                 device: str = 'cuda:1',
                 n_jobs_dataloader: int = 0):

        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size,
                         weight_decay, device, n_jobs_dataloader)
        self.eta = eta
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset, net):
        train_loader, _ = dataset.loaders(batch_size=self.batch_size,
                                          num_workers=self.n_jobs_dataloader)

        print('Hey I am loading net for you!')
        net = net.to(self.device)

        print('Setting hyper-parameters!')
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(net.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.lr_milestones,
                                                   gamma=0.1)

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
                optimizer.zero_grad()
                X_pred = net(X_in)
                dist = criterion(X_pred, X_out)
                dist_mean = torch.mean(dist, axis=[1, 2])
                # The following is the core loss function
                losses = dist_mean
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | '
                        f'Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        print('Training Time: {:.3f}s'.format(self.train_time))
        print('Finished training.')
        return net

    def test(self, dataset, net):
        _, test_loader = dataset.loaders(batch_size=self.batch_size,
                                         num_workers=self.n_jobs_dataloader)
        net = net.to(self.device)
        criterion = nn.MSELoss(reduction='none')

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
                losses = dist_mean
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

        # Compute loss
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        # Log results
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing.')


# --------------------------------------------
# 3.2. (b) Evaluater
# --------------------------------------------
class ForecastEvaluater_(BaseEvaluater):
    def __init__(self,
                 eta: float = 1,
                 batch_size: int = 32,
                 device: str = 'cuda:1',
                 n_jobs_dataloader: int = 0):
        super().__init__(batch_size, device, n_jobs_dataloader)

        # Hyper-parameter for the weight of anomaly training
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
                losses = dist_mean
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

        # Compute loss
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        # Log results
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing.')
