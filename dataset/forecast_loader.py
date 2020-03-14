"""
Title: forecast_loader.py
Description: The loader classes for the forecast model.
Author: Leksai Ye, University of Chicago
"""

from torch.utils.data import DataLoader
from forecast_dataset import ForecastDataset, ForecastDataset_, ForecastDatasetEval


# --------------------------------------------
# 1.2. (a) The Loader Object for Training
# --------------------------------------------
class ForecastLoader:
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
        self.train_set = ForecastDataset(root,
                                         normal_filename,
                                         abnormal_filename,
                                         random_state,
                                         in_size,
                                         out_size,
                                         n_features,
                                         train_portion,
                                         train=True,)
        print('\nHi! I am setting test_set for you.')
        self.test_set = ForecastDataset(root,
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


# --------------------------------------------
# 1.2. (b) The Loader Object for Training (Unsupervised)
# --------------------------------------------
class ForecastLoader_:
    def __init__(self,
                 root: str,
                 normal_filename: str,
                 random_state=42,
                 in_size=100,
                 out_size=25,
                 n_features=128,
                 train_portion=0.8):

        print('Hi! I am setting train_set for you.')
        self.train_set = ForecastDataset_(root,
                                          normal_filename,
                                          random_state,
                                          in_size,
                                          out_size,
                                          n_features,
                                          train_portion,
                                          train=True,)
        print('\nHi! I am setting test_set for you.')
        self.test_set = ForecastDataset_(root,
                                         normal_filename,
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


# --------------------------------------------
# 1.2. (c) The Loader Object for Testing
# --------------------------------------------
class ForecastLoaderEval:
    def __init__(self,
                 root: str,
                 abnormal_filename: str,
                 in_size=100,
                 out_size=25,
                 n_features=128):

        print('Hi! I am setting train_set for you.')
        self.all_set = ForecastDatasetEval(root,
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
