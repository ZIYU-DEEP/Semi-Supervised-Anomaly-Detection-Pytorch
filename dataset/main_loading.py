"""
Title: main_loading.py
Description: The loading functions.
Author: Leksai Ye, University of Chicago
"""

from forecast_loader import ForecastLoader, ForecastLoaderUnsupervised, ForecastLoaderEval
from forecast_loader import DebugLoaderUnsupervised, DebugLoaderEval


def load_dataset(loader_name, root='_', normal_filename='_',
                 abnormal_filename='_', random_state=42, in_size=100,
                 out_size=25, n_features=128, train_portion=0.8):

    known_loaders = ('forecast', 'forecast_unsupervised', 'forecast_eval', 'debug_unsupervised', 'debug_eval')
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
        return ForecastLoaderUnsupervised(root,
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
    if loader_name == 'debug_unsupervised':
        # In this case, root should be '/net/adv_spectrum/torch_data/downtown'
        return DebugLoaderUnsupervised(root)

    if loader_name == 'debug_eval':
        # In this case, root should be folder name
        # e.g. /net/adv_spectrum/torch_data/downtown/abnormal/downtown_LOS-5M-USRP1/file_0
        return DebugLoaderUnsupervised(root)
    return None
