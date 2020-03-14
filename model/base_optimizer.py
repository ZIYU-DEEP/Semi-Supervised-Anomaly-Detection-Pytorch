"""
Title: base_optimizer.py
Description: The base optimizer.
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/optim
"""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Trainer base class.
    """

    def __init__(self,
                 optimizer_name: str,
                 lr: float,
                 n_epochs: int,
                 lr_milestones: tuple,
                 batch_size: int,
                 weight_decay: float,
                 device: str,
                 n_jobs_dataloader: int):

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
    """
    Evaluater base class.
    """

    def __init__(self,
                 batch_size: int,
                 device: str,
                 n_jobs_dataloader: int):

        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader


    @abstractmethod
    def test(self, dataset, net):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass
