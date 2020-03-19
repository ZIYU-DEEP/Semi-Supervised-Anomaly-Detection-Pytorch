"""
Title: main_loading.py
Description: The loading functions.
Author: Lek'Sai Ye, University of Chicago
"""

from deepsad_loader import DeepSADLoader, DeepSADLoaderUnsupervised, DeepSADLoaderEval
from forecast_loader import ForecastLoader, ForecastLoaderUnsupervised, ForecastLoaderEval


def load_dataset(loader_name: str='forecast',
                 root: str='/net/adv_spectrum/torch_data',
                 normal_folder: str='downtown',
                 abnormal_folder: str='downtown_sigOver_10ms'):

    known_loaders = ('forecast', 'forecast_unsupervised', 'forecast_eval',
                     'deepsad', 'deepsad_unsupervised', 'deepsad_eval')
    assert loader_name in known_loaders

    if loader_name == 'forecast':
        return ForecastLoader(root, normal_folder, abnormal_folder)

    if loader_name == 'forecast_unsupervised':
        return ForecastLoaderUnsupervised(root, normal_folder)

    if loader_name == 'forecast_eval':
        # In this case, root should be something different, like:
        # '/net/adv_spectrum/torch_data/downtown/abnormal/downtown_LOS-5M-USRP1/file_0'
        # But no worries, as you do not need to manually specify them
        # main_evaluate.py / main.py will auto-fill in the root
        return ForecastLoaderEval(root)

    if loader_name == 'deepsad':
        return DeepSADLoader(root, normal_folder, abnormal_folder)

    if loader_name == 'deepsad_unsupervised':
        return DeepSADLoaderUnsupervised(root, normal_folder)

    if loader_name == 'deepsad_eval':
        # In this case, root should be something different, like:
        # '/net/adv_spectrum/torch_data/downtown/abnormal/downtown_LOS-5M-USRP1/file_0'
        # But no worries, as you do not need to manually specify them
        # main_evaluate.py / main.py will auto-fill in the root
        return DeepSADLoaderEval(root)

    return None
