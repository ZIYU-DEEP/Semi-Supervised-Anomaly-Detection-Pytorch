"""
Title: main_loading.py
Description: The loading functions.
Author: Leksai Ye, University of Chicago
"""

from forecast_loader import ForecastLoader, ForecastLoader_, ForecastLoaderEval


def load_dataset(loader_name, root='_', normal_filename='_',
                 abnormal_filename='_', random_state=42, in_size=100,
                 out_size=25, n_features=128, train_portion=0.8):

    known_loaders = ('forecast', 'forecast_unsupervised', 'forecast_eval')
    assert loader_name in known_loaders

    if loader_name == 'forecast':
        return ForecastLoader(root,
                              normal_filename,
                              abnormal_filename,
                              random_state,
                              in_size,
                              out_size,
                              n_features,
                              train_portion)

    if loader_name == 'forecast_unsupervised':
        return ForecastLoader_(root,
                               normal_filename,
                               random_state,
                               in_size,
                               out_size,
                               n_features,
                               train_portion)

    if loader_name == 'forecast_eval':
        return ForecastLoaderEval(root,
                                  abnormal_filename,
                                  random_state,
                                  in_size,
                                  out_size,
                                  n_features)
    return None
