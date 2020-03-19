"""
Title: folder_to_xs.py
Description: Preprocessing for spectrum data. You should only use this for normal and sigover.
Author: Lek'Sai Ye, University of Chicago
[Example Command:]
>>> python folder_to_xs_save.py --model deepsad -nf 871 -af 871_ab_sigOver_5ms --train 1
>>> python folder_to_xs_save.py --model deepsad -nf 871 -af 871_ab_sigOver_5ms --train 0
"""

import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='forecast', help='[Choice] forecast, deepsad')
parser.add_argument('-nf', '--normal_folder', type=str, default='ryerson')
parser.add_argument('-af', '--abnormal_folder', type=str, default='',
                    help='do not specify this if no abnormal')
parser.add_argument('--window_size', type=int, default=100, help='Do not try 1000. GPU will die.')
parser.add_argument('--predict_size', type=int, default=25)
parser.add_argument('--n_features', type=int, default=128)
parser.add_argument('--train_portion', type=float, default=0.8)
parser.add_argument('--train', type=int, default=1)
p = parser.parse_args()

model, normal_folder, abnormal_folder = p.model, p.normal_folder, p.abnormal_folder
window_size, predict_size, n_features = p.window_size, p.predict_size, p.n_features
train_portion, train = p.train_portion, p.train

if model == 'deepsad':
    if abnormal_folder:
        root = '/net/adv_spectrum/data/feature/downsample_10/abnormal'
        root_ = '/net/adv_spectrum/torch_data_deepsad'
        path = '{}/{}/{}_{}'.format(root, abnormal_folder, window_size, predict_size)
        path_ = '{}/{}/{}/abnormal/{}'.format(root_, window_size, normal_folder, abnormal_folder)
    else:
        root = '/net/adv_spectrum/data/feature/downsample_10/normal'
        root_ = '/net/adv_spectrum/torch_data_deepsad'
        path = '{}/{}/{}_{}'.format(root, normal_folder, window_size, predict_size)
        path_ = '{}/{}/{}/normal'.format(root_, window_size, normal_folder)
    folder_to_Xs_save_deepsad(path, path_, window_size, n_features, train_portion, train)

elif model == 'forecast':
    if abnormal_folder:
        root = '/net/adv_spectrum/data/feature/downsample_10/abnormal'
        root_ = '/net/adv_spectrum/torch_data'
        path = '{}/{}/{}_{}'.format(root, abnormal_folder, window_size, predict_size)
        path_ = '{}/{}/abnormal/{}'.format(root_, normal_folder, abnormal_folder)
    else:
        root = '/net/adv_spectrum/data/feature/downsample_10/normal'
        root_ = '/net/adv_spectrum/torch_data'
        path = '{}/{}/{}_{}'.format(root, normal_folder, window_size, predict_size)
        path_ = '{}/{}/normal'.format(root_, normal_folder)
    folder_to_Xs_save(path, path_, window_size, n_features, train_portion, train)

print('Done. Got to go to bed. Bye.')
