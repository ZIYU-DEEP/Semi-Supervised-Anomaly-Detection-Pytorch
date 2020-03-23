"""
[Title] main.py
[Description] The main file to run the unsupervised models.
[Author] Lek'Sai Ye, University of Chicago
[Example Command]
> python main.py -ln rec_unsupervised -le rec_eval -nt rec -op rec_unsupervised -nf downtown -af _ -rt /net/adv_spectrum/torch_data_deepsad/100 -gpu 1
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
from main_model_rec import *


# Initialize the parser
parser = argparse.ArgumentParser()
parser.add_argument('--random_state', type=int, default=42)

# Arguments for main_loading
parser.add_argument('-ln', '--loader_name', type=str, default='forecast',
                    help='[Choice]: forecast, ..._unsupervised, deepsad, ..._unsupervised, rec, rec_unsupervised')
parser.add_argument('-le', '--loader_eval_name', type=str, default='forecast_eval',
                    help='forecast_eval, deepsad_eval, rec_eval')
parser.add_argument('-rt', '--root', type=str, default='/net/adv_spectrum/torch_data',
                    help='[Choice]: /net/adv_spectrum/torch_data, /net/adv_spectrum/torch_data_deepsad/100')
parser.add_argument('-nf', '--normal_folder', type=str, default='downtown',
                    help='[Example]: downtown, ryerson_train, campus_drive')
parser.add_argument('-af', '--abnormal_folder', type=str, default='downtown_sigOver_10ms',
                    help='[Example]: _, downtown_sigOver_10ms, downtown_sigOver_5ms')

# Arguments for main_network
parser.add_argument('-nt', '--net_name', type=str, default='lstm_stacked',
                    help='[Choice]: lstm, lstm_stacked, lstm_autoencoder, rec')
parser.add_argument('-rp', '--rep_dim', type=int, default=10,
                    help='Only apply to DeepSAD model - the latent dimension.')

# Arguments for main_model
parser.add_argument('-pt', '--pretrain', type=bool, default=True,
                    help='[Choice]: Only apply to DeepSAD model: True, False')
parser.add_argument('--load_model', type=str, default='',
                    help='[Example]: ./deepsad_ryerson_train_ryerson_ab_train_sigOver_10ms/net_lstm_encoder_eta_100_epochs_100_batch_128/model.tar')
parser.add_argument('-op', '--optimizer_', type=str, default='forecast_exp',
                    help='[Choice]: forecast_unsupervised, forecast_exp, forecast_minus, rec, rec_unsupervised')
parser.add_argument('-et', '--eta_str', default=100,
                    help='The _% representation of eta - choose from 100, 50, 25, etc.')
parser.add_argument('--optimizer_name', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--ae_n_epochs', type=int, default=100)
parser.add_argument('--lr_milestones', type=str, default='50_100_150')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('-gpu', '--device_no', type=int, default=1)
parser.add_argument('--n_jobs_dataloader', type=int, default=0)
parser.add_argument('--save_ae', type=bool, default=True,
                    help='Only apply to Deep SAD model.')
parser.add_argument('--load_ae', type=bool, default=False,
                    help='Only apply to Deep SAD model.')

# Arguments for output_paths
parser.add_argument('--txt_filename', type=str, default='full_results.txt')
p = parser.parse_args()

# Extract the arguments
random_state, loader_name, loader_eval_name = p.random_state, p.loader_name, p.loader_eval_name
root, normal_folder, abnormal_folder = p.root, p.normal_folder, p.abnormal_folder
net_name, rep_dim, pretrain, load_model = p.net_name, p.rep_dim, p.pretrain, p.load_model
optimizer_, eta_str, optimizer_name = p.optimizer_, p.eta_str, p.optimizer_name
lr, n_epochs, ae_n_epochs, batch_size = p.lr, p.n_epochs, p.ae_n_epochs, p.batch_size
lr_milestones = tuple(int(i) for i in p.lr_milestones.split('_'))
weight_decay, device_no, n_jobs_dataloader = p.weight_decay, p.device_no, p.n_jobs_dataloader
save_ae, load_ae = p.save_ae, p.load_ae
txt_filename = p.txt_filename

# Define folder to save the model and relating results
if loader_eval_name == 'forecast_eval':
    folder_name = '{}_{}_{}'.format(optimizer_, normal_folder, abnormal_folder)
    out_path = '/net/adv_spectrum/torch_models/forecast/{}'.format(folder_name)
    final_path = '{}/net_{}_eta_{}_epochs_{}_batch_{}'.format(out_path, net_name, eta_str,
                                                              n_epochs, batch_size)
elif loader_eval_name == 'rec_eval':
    folder_name = '{}_{}_{}'.format(optimizer_, normal_folder, abnormal_folder)
    out_path = '/net/adv_spectrum/torch_models/rec/{}'.format(folder_name)
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
results_path = Path(final_path) / 'results.json'
ae_results_path = Path(final_path) / 'ae_results.json'
result_df_path = Path(final_path) / 'result_df.pkl'
cut_95_path = Path(final_path) / 'cut_95.npy'
cut_99_path = Path(final_path) / 'cut_99.npy'

# Define additional stuffs
device = 'cuda:{}'.format(device_no)
eta = float(eta_str * 0.01)

# Set random state
torch.manual_seed(random_state)

#############################################
# 1. Model Training
#############################################
# Initialize data
dataset = load_dataset(loader_name, root, normal_folder, abnormal_folder)

# Load Deep SAD model
if loader_name in ['deepsad', 'deepsad_unsupervised']:
    # Define model
    model = DeepSADModel(optimizer_, eta)
    model.set_network(net_name)

    # Load other models if specified
    if load_model:
        print('Loading model from {}'.format(load_model))
        model.load_model(model_path=load_model,
                         load_ae=True,
                         map_location=device)
    # Pretrain if specified
    if pretrain:
        print('I am pre-training for you.')
        model.pretrain(dataset, optimizer_name, lr, ae_n_epochs, lr_milestones,
                       batch_size, weight_decay, device, n_jobs_dataloader)
        model.save_ae_results(export_json=ae_results_path)

# Load Forecast model
elif loader_name in ['forecast', 'forecast_unsupervised']:
    model = ForecastModel(optimizer_, eta)
    model.set_network(net_name)

elif loader_name in ['rec', 'rec_unsupervised']:
    model = RecModel(optimizer_, eta)
    model.set_network(net_name)

# Training model
model.train(dataset, eta, optimizer_name, lr, n_epochs, lr_milestones,
            batch_size, weight_decay, device, n_jobs_dataloader)


#############################################
# 2. Model Testing
#############################################
# Test and Save model
model.test(dataset, device, n_jobs_dataloader)
model.save_results(export_json=results_path)
model.save_model(export_model=model_path, save_ae=save_ae)

# Prepare to write the results
indices, labels, scores = zip(*model.results['test_scores'])
indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)

result_df = pd.DataFrame()
result_df['indices'] = indices
result_df['labels'] = labels
result_df['scores'] = scores
result_df.to_pickle(result_df_path)

result_df.drop('indices', inplace=True, axis=1)
df_normal = result_df[result_df.labels == 0]
df_abnormal = result_df[result_df.labels == 1]

# Save the threshold
cut_95 = df_normal.scores.quantile(0.95)
y_95 = [1 if e > cut_95 else 0 for e in df_abnormal['scores'].values]
np.save(cut_95_path, cut_95)

cut_99 = df_normal.scores.quantile(0.99)
y_99 = [1 if e > cut_99 else 0 for e in df_abnormal['scores'].values]
np.save(cut_99_path, cut_99)


# Write the basic test file
f = open(txt_result_file, 'a')
f.write('############################################################\n')
f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
f.write('\n[DataFrame Name] {}\n'.format(result_df_path))
f.write('[Normal Folder] {}\n'.format(normal_folder))
f.write('[Abnormal Filename] {}\n'.format(abnormal_folder))
f.write('[Model] {}\n'.format(optimizer_))
f.write('[Eta] {}\n'.format(eta))
f.write('[Epochs] {}\n'.format(n_epochs))
f.write('[Cut Threshold with 0.05 FP Rate] {}\n'.format(cut_95))
f.write('[Cut Threshold with 0.01 FP Rate] {}\n'.format(cut_99))
if len(df_abnormal):
    f.write('[A/N Ratio] 1:{}\n'.format(len(df_abnormal) / len(df_normal)))
    f.write('[Recall for {} (FP = 0.05)] {}\n'.format('TEST', sum(y_95) / len(y_95)))
    f.write('[Recall for {} (FP = 0.01)] {}\n'.format('TEST', sum(y_99) / len(y_99)))
f.write('---------------------\n')
f.close()


#############################################
# 3. Model Evaluation
#############################################
f = open(txt_result_file, 'a')

l_root_abnormal = ['/net/adv_spectrum/{}/{}/abnormal/{}_sigOver_5ms',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_sigOver_10ms',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_sigOver_20ms',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_LOS-5M-USRP1',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_LOS-5M-USRP2',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_LOS-5M-USRP3',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_NLOS-5M-USRP1',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_Dynamics-5M-USRP1',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_wn_1.4G',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_wn_5G',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_fsk_1.4G',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_fsk_5G',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_psk_1.4G',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_psk_5G',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_qam_1.4G',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_qam_5G',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_ofdm_1.4G',
                   '/net/adv_spectrum/{}/{}/abnormal/{}_ofdm_5G']

for root_abnormal in l_root_abnormal:
    # No bugs please.
    print('I am starting evaluation for you.')
    print('Abracadabra! Prajnaparamita! JI-JI-RU-LV-LING!')

    # Formating the path
    if loader_eval_name in ['forecast_eval']: mid_root = 'torch_data'
    else: mid_root = 'torch_data_deepsad/100'

    if normal_folder == 'ryerson_train': normal_folder_ = 'ryerson_ab_train'
    elif normal_folder == '871': normal_folder_ = '871_ab'
    else: normal_folder_ = normal_folder

    root_abnormal = root_abnormal.format(mid_root, normal_folder, normal_folder_)

    # if root_abnormal in ['/net/adv_spectrum/torch_data/871/abnormal/871_ab_sigOver_10ms',
    #                      '/net/adv_spectrum/torch_data/871/abnormal/871_ab_sigOver_20ms',
    #                      '/net/adv_spectrum/torch_data_deepsad/100/871/abnormal/871_ab_sigOver_10ms',
    #                      '/net/adv_spectrum/torch_data_deepsad/100/871/abnormal/871_ab_sigOver_20ms',]:
    #     continue

    if not os.path.exists(root_abnormal):
        print('Skip {}!'.format(root_abnormal))
        continue

    f.write('============================================================\n')
    f.write('Results for {}:\n'.format(root_abnormal))

    # Start evaluating
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
        elif loader_eval_name in ['rec_eval']:
            model_eval = RecModelEval(optimizer_, eta=eta)

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
