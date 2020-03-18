"""
Title: forecast_loader.py
Description: The loader classes for the forecast model.
Author: Lek'Sai Ye, University of Chicago
"""

from torch.utils.data import DataLoader
from forecast_dataset import ForecastDataset, ForecastDatasetUnsupervised, ForecastDatasetEval


# --------------------------------------------
# Loader for Semisupervised Forecast Model (root, abnormal_filename)
# --------------------------------------------
class ForecastLoader:
    def __init__(self,
                 root: str='/net/adv_spectrum/torch_data',
                 normal_folder: str='downtown',
                 abnormal_folder: str='downtown_sigOver_10ms'):

        print('Hi! I am setting trainning set for you.')
        self.train_set = ForecastDataset(root,
                                         normal_folder,
                                         abnormal_folder,
                                         train=1)
        print('\nHi! I am setting testing set for you.')
        self.test_set = ForecastDataset(root,
                                        normal_folder,
                                        abnormal_folder,
                                        train=0)
    def loaders(self,
                batch_size: int=128,
                shuffle_train: bool=True,
                shuffle_test: bool=False,
                num_workers: int=0) -> (DataLoader, DataLoader):

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


# --------------------------------------------
# Loader for Unsupervised Forecast Model (root)
# --------------------------------------------
class ForecastLoaderUnsupervised:
    def __init__(self,
                 root: str='/net/adv_spectrum/torch_data',
                 normal_folder: str='downtown'):

        print('Hi! I am setting train_set for you.')
        self.train_set = ForecastDatasetUnsupervised(root,
                                                     normal_folder,
                                                     True)
        print('\nHi! I am setting test_set for you.')
        self.test_set = ForecastDatasetUnsupervised(root,
                                                     normal_folder,
                                                     False)

    def loaders(self,
                batch_size: int=128,
                shuffle_train: bool=True,
                shuffle_test: bool=False,
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


# --------------------------------------------
# Loader to Evaluate Forecast Model (folder)
# --------------------------------------------
class ForecastLoaderEval:
    def __init__(self,
                 root: str='/net/adv_spectrum/torch_data/downtown/abnormal/downtown_LOS-5M-USRP1/file_0'):

        print('Hi! I am setting all set for you.')
        self.all_set = ForecastDatasetEval(root)

    def loaders(self,
                batch_size: int=128,
                shuffle_all: bool=False,
                num_workers: int=0) -> (DataLoader, DataLoader):

        all_loader = DataLoader(dataset=self.all_set,
                                batch_size=batch_size,
                                shuffle=shuffle_all,
                                num_workers=num_workers,
                                drop_last=True)
        return all_loader

    def __repr__(self):
        return self.__class__.__name__
