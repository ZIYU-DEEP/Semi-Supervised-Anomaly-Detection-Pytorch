"""
Title: save_abnormal_xs.py
Description: Preprocessing for spectrum data. You should only use this for anomalies.
Author: Lek'Sai Ye, University of Chicago
[Example Command:]
>>> python save_abnormal_xs.py --model deepsad -nf 871 -af 871_ab_sigOver_5ms
>>> python save_abnormal_xs.py --model deepsad -nf 871 -af 871_ab_LOS-5M-USRP1
>>> python save_abnormal_xs.py --model deepsad -nf 871 -af 871_ab_LOS-5M-USRP2
>>> python save_abnormal_xs.py --model deepsad -nf 871 -af 871_ab_LOS-5M-USRP3
>>> python save_abnormal_xs.py --model deepsad -nf 871 -af 871_ab_NLOS-5M-USRP1
>>> python save_abnormal_xs.py --model deepsad -nf 871 -af 871_ab_Dynamics-5M-USRP1
"""

import argparse
from pathlib import Path
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='forecast', help='[Choice] forecast, deepsad')
parser.add_argument('-nf', '--normal_folder', type=str, default='871', help='The prefix of anomaly.')
parser.add_argument('-af', '--abnormal_folder', type=str, default='871_ab_LOS-5M-USRP1')
parser.add_argument('--window_size', type=int, default=100, help='Do not try 1000. GPU will die.')
parser.add_argument('--predict_size', type=int, default=25)
parser.add_argument('--n_features', type=int, default=128)
p = parser.parse_args()

model, normal_folder, abnormal_folder = p.model, p.normal_folder, p.abnormal_folder
window_size, predict_size, n_features = p.window_size, p.predict_size, p.n_features

if model == 'deepsad':
    root = '/net/adv_spectrum/data/feature/downsample_10/abnormal'
    root_ = '/net/adv_spectrum/torch_data_deepsad'
    path = '{}/{}/{}_{}'.format(root, abnormal_folder, window_size, predict_size)
    path_ = '{}/{}/{}/abnormal/{}'.format(root_, window_size, normal_folder, abnormal_folder)
    save_abnormal_Xs_deepsad(path, path_, window_size, n_features)

elif model == 'forecast':
    root = '/net/adv_spectrum/data/feature/downsample_10/abnormal'
    root_ = '/net/adv_spectrum/torch_data'
    path = '{}/{}/{}_{}'.format(root, abnormal_folder, window_size, predict_size)
    path_ = '{}/{}/abnormal/{}'.format(root_, normal_folder, abnormal_folder)
    save_abnormal_Xs(path, path_, window_size, n_features)

print('Done. Got to go to bed. Bye.')
