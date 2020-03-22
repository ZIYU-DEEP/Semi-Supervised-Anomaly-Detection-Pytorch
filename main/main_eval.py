"""
[Title] main.py
[Description] The main file to run the unsupervised models.
[Author] Lek'Sai Ye, University of Chicago
[Example Command]
> For <supervised training>:
>>> python main.py -nf ryerson_train -af ryerson_ab_train_sigOver_10ms -gpu 2
>>> python main.py -nf downtown -af downtown_sigOver_10ms -gpu 2
>>> python main.py -nf campus_drive -af campus_drive_sigOver_10ms -gpu 2
>>> python main.py -nf 871 -af 871_ab_sigOver_5ms -gpu 2

> For <unsupervised training>:
>>> python main.py -ln forecast_unsupervised -op forecast_unsupervised -nf ryerson_train -gpu 1
>>> python main.py -ln forecast_unsupervised -op forecast_unsupervised -nf downtown -gpu 3
>>> python main.py -ln forecast_unsupervised -op forecast_unsupervised -nf campus_drive -gpu 3
>>> python main.py -ln forecast_unsupervised -op forecast_unsupervised -nf 871 -gpu 2
"""

#############################################
# 0. Preparation
#############################################
import sys
sys.path.append('../utils/')
sys.path.append('../dataset/')
sys.path.append('../network/')
sys.path.append('../model/')

import os
import glob
import time
import torch
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from main_loading import *
from main_network import *
from main_model_forecast import *
from main_model_deepsad import *


# Initialize the parser
parser = argparse.ArgumentParser()
parser.add_argument('--random_state', type=int, default=42)

# Arguments for main_loading
parser.add_argument('-le', '--loader_eval_name', type=str, default='forecast_eval',
                    help='forecast_eval, deepsad_eval')
parser.add_argument('-rt', '--root', type=str, default='/net/adv_spectrum/torch_data',
                    help='[Choice]: .../torch_data, .../torch_data_deepsad/100')
parser.add_argument('-nf', '--normal_folder', type=str, default='downtown',
                    help='[Example]: downtown, ryerson_train, campus_drive')
parser.add_argument('-af', '--abnormal_folder', type=str, default='downtown_sigOver_10ms',
                    help='[Example]: _, downtown_sigOver_10ms, downtown_sigOver_5ms')

# Arguments for main_network
parser.add_argument('--net_name', type=str, default='lstm_stacked',
                    help='[Choice]: lstm, lstm_stacked, lstm_autoencoder')

# Arguments for main_model
parser.add_argument('-pt', '--pretrain', type=bool, default=True,
                    help='[Choice]: Only apply to DeepSAD model: True, False')
parser.add_argument('-op', '--optimizer_', type=str, default='forecast_exp',
                    help='[Choice]: forecast_unsupervised, forecast_exp, forecast_minus')
parser.add_argument('-et', '--eta_str', default=100,
                    help='The _% representation of eta - choose from 100, 50, 25, etc.')
parser.add_argument('--optimizer_name', type=str, default='adam')
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('-gpu', '--device_no', type=int, default=1)
parser.add_argument('--n_jobs_dataloader', type=int, default=0)

# Arguments for output_paths
parser.add_argument('--txt_filename', type=str, default='full_results.txt')
p = parser.parse_args()

# Extract the arguments
random_state, loader_eval_name = p.random_state, p.loader_eval_name
root, normal_folder, abnormal_folder = p.root, p.normal_folder, p.abnormal_folder
net_name, pretrain = p.net_name, p.pretrain
optimizer_, eta_str, optimizer_name = p.optimizer_, p.eta_str, p.optimizer_name
n_epochs, batch_size = p.n_epochs, p.batch_size
device_no, n_jobs_dataloader = p.device_no, p.n_jobs_dataloader
txt_filename = p.txt_filename

# Define folder to save the model and relating results
if loader_eval_name == 'forecast_eval':
    folder_name = '{}_{}_{}'.format(optimizer_, normal_folder, abnormal_folder)
    out_path = '/net/adv_spectrum/torch_models/forecast/{}'.format(folder_name)
    final_path = '{}/net_{}_eta_{}_epochs_{}_batch_{}'.format(out_path, net_name, eta_str,
                                                              n_epochs, batch_size)

elif loader_eval_name == 'deepsad_eval':
    folder_name = '{}_{}_{}_{}'.format(optimizer_, str(pretrain), normal_folder, abnormal_folder)
    out_path = '/net/adv_spectrum/torch_models/deepsad/{}'.format(folder_name)
    final_path = '{}/net_{}_eta_{}_epochs_{}_batch_{}'.format(out_path, net_name, eta_str,
                                                              n_epochs, batch_size)
if not os.path.exists(out_path): os.makedirs(out_path)
if not os.path.exists(final_path): os.makedirs(final_path)

# Define the general txt file path with stores all models' results
txt_result_file = '{}/{}'.format(out_path, txt_filename)

# Define the path for others
model_path = Path(final_path) / 'model.tar'
cut_95_path = Path(final_path) / 'cut_95.npy'
cut_99_path = Path(final_path) / 'cut_99.npy'

# Define additional stuffs
device = 'cuda:{}'.format(device_no)
eta = float(eta_str * 0.01)
cut_95 = float(np.load(cut_95_path))
cut_99 = float(np.load(cut_99_path))

# Set random state
torch.manual_seed(random_state)

#############################################
# 3. Model Evaluation
#############################################
f = open(txt_result_file, 'a')

l_root_abnormal = ['/net/adv_spectrum/torch_data/{}/abnormal/{}_sigOver_5ms',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_sigOver_10ms',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_sigOver_20ms',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_LOS-5M-USRP1',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_LOS-5M-USRP2',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_LOS-5M-USRP3',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_NLOS-5M-USRP1',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_Dynamics-5M-USRP1',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_wn_1.4G',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_wn_5G',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_fsk_1.4G',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_fsk_5G',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_psk_1.4G',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_psk_5G',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_qam_1.4G',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_qam_5G',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_ofdm_1.4G',
                   '/net/adv_spectrum/torch_data/{}/abnormal/{}_ofdm_5G']

for root_abnormal in l_root_abnormal:
    print('I am starting evaluation for you.')
    print('Abracadabra! Prajnaparamita! JI-JI-RU-LV-LING!')

    if normal_folder == 'ryerson_train': normal_folder_ = 'ryerson_ab_train'
    elif normal_folder == '871': normal_folder_ = '871_ab'
    else: normal_folder_ = normal_folder

    root_abnormal = root_abnormal.format(normal_folder, normal_folder_)

    if root_abnormal in ['/net/adv_spectrum/torch_data/871/abnormal/871_ab_sigOver_10ms',
                         '/net/adv_spectrum/torch_data/871/abnormal/871_ab_sigOver_20ms']:
        continue

    f.write('============================================================\n')
    f.write('Results for {}:\n'.format(root_abnormal))

    total_recall_95 = []
    total_recall_99 = []
    for i, folder in enumerate(sorted(glob.glob(root_abnormal + '/file*'))):
        # Let everyone know which file I am processing
        print(folder)

        # Load dataset for evaluation
        dataset_eval = load_dataset(loader_eval_name, folder)

        # Load model for evaluation
        if loader_eval_name in ['forecast_eval']:
            model_eval = ForecastModelEval(optimizer_, eta=eta)
        elif loader_eval_name in ['deepsad_eval']:
            model_eval = DeepSADModelEval(optimizer_, eta=eta)

        model_eval.set_network(net_name)
        model_eval.load_model(model_path=model_path, map_location=device)

        # Test the model
        model_eval.test(dataset_eval,
                        eta=eta,
                        batch_size=batch_size,
                        device=device,
                        n_jobs_dataloader=n_jobs_dataloader)

        _, _, scores = zip(*model_eval.results['test_scores'])
        f.write('---------------------\n')

        # Record results when FP = 0.05
        y_95 = [1 if e > cut_95 else 0 for e in scores]
        recall_95 = sum(y_95) / len(y_95)
        total_recall_95.append(recall_95)
        f.write('[Recall for file {} (FP = 0.05)] {}\n'.format(i, recall_95))

        # Record results when FP = 0.01
        y_99 = [1 if e > cut_99 else 0 for e in scores]
        recall_99 = sum(y_99) / len(y_99)
        total_recall_99.append(recall_99)
        f.write('[Recall for file {} (FP = 0.01)] {}\n'.format(i, recall_99))

    # Save averaged results when FP = 0.05
    mean_recall_95 = np.array(total_recall_95).mean()
    std_recall_95 = np.array(total_recall_95).std()
    f.write('---------------------\n')
    f.write('[FP rate] 0.05')
    f.write('\n[**Recall Mean**] {}\n[**Recall std**] {}\n\n'.format(mean_recall_95, std_recall_95))
    print('\n[**Recall Mean**] {}\n[**Recall std**] {}\n'.format(mean_recall_95, std_recall_95))

    # Save averaged results when FP = 0.01
    mean_recall_99 = np.array(total_recall_99).mean()
    std_recall_99 = np.array(total_recall_99).std()
    f.write('---------------------\n')
    f.write('[FP rate] 0.01')
    f.write('\n[**Recall Mean**] {}\n[**Recall std**] {}\n\n'.format(mean_recall_99, std_recall_99))
    print('\n[**Recall Mean**] {}\n[**Recall std**] {}\n'.format(mean_recall_99, std_recall_99))

f.write('###########################################################\n\n\n\n')
f.close()
print('Finished. Now I am going to bed. Bye.')
