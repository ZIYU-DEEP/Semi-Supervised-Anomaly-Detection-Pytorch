
"""
Title: main_model_deepsad.py
Description: The main classes for models.
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/DeepSAD.py
"""

import sys
import torch
import json
from deepsad_optimizer import DeepSADTrainer,DeepSADTrainer_, DeepSADEvaluater, AETrainer
# from deepsad_unsupervised_optimizer import DeepSADTrainer_, DeepSADEvaluater_
sys.path.append('../network/')
from main_network import build_network, build_autoencoder


# --------------------------------------------
# 3.3. (a) Model Object for training
# --------------------------------------------
class DeepSADModel:
    def __init__(self,
                 optimizer_: str = 'deepsad_exp',
                 eta: float = 1.0):
        known_optimizer_ = ('deepsad', 'deepsad_unsupervised')
        assert optimizer_ in known_optimizer_

        self.optimizer_ = optimizer_
        self.c = None
        self.eta = eta
        self.net_name = None

        self.net = None
        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None
        self.ae_trainer = None
        self.optimizer_name = None

        self.results = {'train_time': None, 'test_auc': None,
                        'test_time': None, 'test_scores': None}
        self.ae_results = {'train_time': None, 'test_auc': None,
                           'test_time': None}

    def set_network(self,
                    net_name: str='lstm_encoder'):
        """
        Set the network structure for the DeepSAD model.
        The key here is to initialize <self.net>.
        """
        self.net_name = net_name
        # Note that <build_network> is different from <build_autoencoder>
        # Yet they share the same parameter
        self.net = build_network(net_name)

    def load_model(self,
                   model_path,
                   load_ae=False,
                   map_location='cuda:1'):
        """
        Load the trained model for the DeepSAD model.
        The key here is to initialize <self.c>.
        """
        # Load the general DeepSAD model
        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        # Load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def init_network_weights_from_pretraining(self):
        """
        If pretraining is specified, we will load the networks
        from the pretrained ae net.
        """
        # Obtain the net dictionary
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}

        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)

        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def train(self,
              dataset,
              eta: float=1.0,
              optimizer_name: str='adam',
              lr: float=0.001,
              n_epochs: int=60,
              lr_milestones: tuple=(100, 160, 220),
              batch_size: int=32,
              weight_decay: float=1e-6,
              device: str='cuda:1',
              n_jobs_dataloader: int=0):
        print('Learning rate: {}'.format(lr))
        self.optimizer_name = optimizer_name

        if self.optimizer_ == 'deepsad':
            self.trainer = DeepSADTrainer(self.c,
                                          self.eta,
                                          optimizer_name,
                                          lr,
                                          n_epochs,
                                          lr_milestones,
                                          batch_size,
                                          weight_decay,
                                          device,
                                          n_jobs_dataloader)
        if self.optimizer_ == 'deepsad_unsupervised':
            self.trainer = DeepSADTrainer_(self.c,
                                           self.eta,
                                           optimizer_name,
                                           lr,
                                           n_epochs,
                                           lr_milestones,
                                           batch_size,
                                           weight_decay,
                                           device,
                                           n_jobs_dataloader)

        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()

    def test(self,
             dataset,
             device: str = 'cuda:1',
             n_jobs_dataloader: int = 0):
        if self.trainer is None:
            if self.optimizer_ == 'deepsad':
                self.trainer = DeepSADTrainer(self.c,
                                              self.eta,
                                              device=device,
                                              n_jobs_dataloader=n_jobs_dataloader)
            if self.optimizer_ == 'deepsad_unsupervised':
                self.trainer = DeepSADTrainer_(self.c,
                                               self.eta,
                                               device=device,
                                               n_jobs_dataloader=n_jobs_dataloader)
        self.trainer.test(dataset, self.net)

        if self.trainer.test_auc:
            self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self,
                 dataset,
                 optimizer_name: str='adam',
                 lr: float=0.001,
                 n_epochs: int=100,
                 lr_milestones: tuple=(50, 100, 150, 200),
                 batch_size: int=128,
                 weight_decay: float=1e-6,
                 device: str='cuda:1',
                 n_jobs_dataloader: int=0):
        # Set autoencoder network
        # Note that <build_autoencoder> is different from <build_network>
        # Yet they share the same parameter
        self.ae_net = build_autoencoder(self.net_name)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name,
                                    lr=lr,
                                    n_epochs=n_epochs,
                                    lr_milestones=lr_milestones,
                                    batch_size=batch_size,
                                    weight_decay=weight_decay,
                                    device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net)

        # Get test results
        # self.ae_results['test_auc'] = self.ae_trainer.test_auc
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def save_model(self, export_model, save_ae=True):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if (save_ae and self.ae_net is not None) else None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def save_results(self, export_json):
        with open(export_json, 'w') as f:
            json.dump(self.results, f)

    def save_ae_results(self, export_json):
        with open(export_json, 'w') as f:
            json.dump(self.ae_results, f)


# --------------------------------------------
# 3.3. (b) Model Object for Evaluating
# --------------------------------------------
class DeepSADModelEval:
    def __init__(self,
                 optimizer_,
                 eta: float = 1.0):
        known_optimizer_ = ('deepsad', 'deepsad_unsupervised')
        assert optimizer_ in known_optimizer_
        self.optimizer_ = optimizer_
        self.eta = eta
        self.net_name = None
        self.net = None
        self.evaluater= None
        self.optimizer_name = None
        self.results = {'test_time': None,'test_scores': None}

    def set_network(self, net_name):
        """
        The key here is to initialize <self.net>.
        """
        self.net_name = net_name
        self.net = build_network(net_name)

    def load_model(self, model_path, map_location='cuda:1'):
        """
        The key here is to fill in <self.c> and <self.net>.
        """
        model_dict = torch.load(model_path, map_location=map_location)
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

    def test(self,
             dataset,
             eta: float=1,
             batch_size: int=128,
             device: str='cuda:1',
             n_jobs_dataloader: int = 0):

        if self.evaluater is None:
            self.evaluater = DeepSADEvaluater(self.c,
                                              self.eta,
                                              batch_size=batch_size,
                                              device=device,
                                              n_jobs_dataloader=n_jobs_dataloader)

        self.evaluater.test(self.optimizer_,
                            dataset,
                            self.net)
        self.results['test_time'] = self.evaluater.test_time
        self.results['test_scores'] = self.evaluater.test_scores

    def save_results(self, export_json):
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
