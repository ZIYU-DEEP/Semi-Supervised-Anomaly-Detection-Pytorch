"""
[Title] main.py
[Description] The main file to run the unsupervised models.
[Author] Lek'Sai Ye, University of Chicago
[Example Command:]
> For <supervised training>:
>>> python main_deepsad.py -nf ryerson_train -af ryerson_ab_train_sigOver_10ms

> For <unsupervised training>:
>>> python main_deepsad.py -ln deepsad_unsupervised -op deepsad_unsupervised
                           -nf ryerson_train -af ryerson_ab_train_sigOver_10ms
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
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from main_loading import *
from main_network import *
from main_model_forecast import *
from main_model_deepsad import *

# Example Command:
# python main_deepsad.py --loader_name deepsad_unsupervised --optimizer_ deepsad_unsupervised

# Initialize the parser
parser = argparse.ArgumentParser()
parser.add_argument('--random_state', type=int, default=42)

# Arguments for main_loading
parser.add_argument('-ln', '--loader_name', type=str, default='deepsad',
                    help='[Choice]: deepsad, deepsad_unsupervised')
parser.add_argument('--loader_eval_name', type=str, default='deepsad_eval')
parser.add_argument('--root', type=str, default='/net/adv_spectrum/torch_data_deepsad/100',
                    help='[Choice]: /net/adv_spectrum/torch_data_deepsad')
parser.add_argument('-nf', '--normal_folder', type=str, default='ryerson_train',
                    help='[Example]: downtown, ryerson_train, campus_drive, 871')
parser.add_argument('-af', '--abnormal_folder', type=str, default='ryerson_ab_train_sigOver_10ms',
                    help='[Example]: _, downtown_sigOver_10ms, downtown_sigOver_5ms')

# Arguments for main_network
parser.add_argument('--net_name', type=str, default='lstm_autoencoder',
                    help='[Choice]: lstm_autoencoder')
parser.add_argument('-rp', '--rep_dim', type=int, default=10,
                    help='Only apply to DeepSAD model - the latent dimension.')

# Arguments for main_model
parser.add_argument('-pt', '--pretrain', type=bool, default=True,
                    help='[Choice]: Only apply to DeepSAD model: True, False')
parser.add_argument('--load_model', type=str, default='',
                    help='[Example]: ./deepsad_ryerson_train_ryerson_ab_train_sigOver_10ms/net_lstm_encoder_eta_100_epochs_100_batch_128/model.tar')
parser.add_argument('-op', '--optimizer_', type=str, default='deepsad',
                    help='[Choice]: deepsad, deepsad_unsupervised')
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
parser.add_argument('--load_ae', type=bool, default=True,
                    help='Only apply to Deep SAD model.')
parser.add_argument('--fp_rate', type=float, default=0.05,
                    help='The false positive rate as the judge threshold.')

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
save_ae, load_ae, fp_rate = p.save_ae, p.load_ae, p.fp_rate
txt_filename = p.txt_filename

# Define folder to save the model and relating results
folder_name = '{}_{}_{}_{}'.format(optimizer_, str(pretrain), normal_folder, abnormal_folder)
out_path = '../results/{}'.format(folder_name)  # change '.' to '/net/adv_spectrum/torch_model' in future
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
cut_path = Path(final_path) / 'cut.pkl'

# Define additional stuffs
device = 'cuda:{}'.format(device_no)
eta = float(eta_str * 0.01)

# Set random state
torch.manual_seed(random_state)

#############################################
# 1. Model Training
#############################################
# Load data
dataset = load_dataset(loader_name, root, normal_folder, abnormal_folder)

# Initialize model
if loader_name in ['forecast', 'forecast_unsupervised']:
    model = ForecastModel(optimizer_, eta)
    model.set_network(net_name)
if loader_name in ['deepsad', 'deepsad_unsupervised']:
    model = DeepSADModel(optimizer_, eta)
    model.set_network(net_name)

# Load model if specified
if load_model:
    print('Loading model from {}'.format(load_model))
    model.load_model(model_path=load_model,
                     load_ae=True,
                     map_location=device)

if pretrain:
    print('I am pre-training for you.')
    model.pretrain(dataset,
                   optimizer_name,
                   lr,
                   ae_n_epochs,
                   lr_milestones,
                   batch_size,
                   weight_decay,
                   device,
                   n_jobs_dataloader)
    model.save_ae_results(export_json=ae_results_path)

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
cut = df_normal.scores.quantile(1 - fp_rate)
y = [1 if e > cut else 0 for e in df_abnormal['scores'].values]
np.save(cut_path, cut)

# Write the basic test file
f = open(txt_result_file, 'a')
f.write('############################################################\n')
f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
f.write('[DataFrame Name] {}\n'.format(result_df_path))
f.write('[Normal Folder] {}\n'.format(normal_folder))
f.write('[Abnormal Filename] {}\n'.format(abnormal_folder))
f.write('[Model] {}\n'.format(optimizer_))
f.write('[Eta] {}\n'.format(eta))
f.write('[Epochs] {}\n'.format(n_epochs))
f.write('[False Positive Rate] {}\n'.format(fp_rate))
f.write('[Cut Threshold] {}\n'.format(cut))
if len(df_abnormal):
    f.write('[A/N Ratio] 1:{}\n'.format(len(df_abnormal) / len(df_normal)))
    f.write('[Train AUC] {}\n'.format(model.results['test_auc']))
    f.write('[Recall for {}] {}\n'.format('TEST', sum(y) / len(y)))
f.write('---------------------\n')


#############################################
# 3. Model Evaluation
#############################################
l_root_abnormal = ['/net/adv_spectrum/torch_data_deepsad/100/{}/abnormal/{}_sigOver_5ms',
                   '/net/adv_spectrum/torch_data_deepsad/100/{}/abnormal/{}_sigOver_10ms',
                   '/net/adv_spectrum/torch_data_deepsad/100/{}/abnormal/{}_sigOver_20ms',
                   '/net/adv_spectrum/torch_data_deepsad/100/{}/abnormal/{}_LOS-5M-USRP1',
                   '/net/adv_spectrum/torch_data_deepsad/100/{}/abnormal/{}_LOS-5M-USRP2',
                   '/net/adv_spectrum/torch_data_deepsad/100/{}/abnormal/{}_LOS-5M-USRP3',
                   '/net/adv_spectrum/torch_data_deepsad/100/{}/abnormal/{}_NLOS-5M-USRP1',
                   '/net/adv_spectrum/torch_data_deepsad/100/{}/abnormal/{}_Dynamics-5M-USRP1']

for root_abnormal in l_root_abnormal:
    print('I am starting evaluation for you.')
    print('Abracadabra! Prajnaparamita! JI-JI-RU-LV-LING!')

    if normal_folder == 'ryerson_train': normal_folder_ = 'ryerson_ab_train'
    elif normal_folder == '871': normal_folder_ = '871_ab'
    else: normal_folder_ = normal_folder

    root_abnormal = root_abnormal.format(normal_folder, normal_folder_)

    if root_abnormal in ['/net/adv_spectrum/torch_data_deepsad/100/871/abnormal/871_ab_sigOver_10ms',
                         '/net/adv_spectrum/torch_data_deepsad/100/871/abnormal/871_ab_sigOver_20ms']:
        continue

    f.write('============================================================\n')
    f.write('Results for {}:\n'.format(root_abnormal))
    total_recall = []
    for i, folder in enumerate(sorted(glob.glob(root_abnormal + '/file*'))):
        print(folder)
        # Load dataset for evaluation
        dataset_eval = load_dataset(loader_eval_name, folder)
        # Load model for evaluation
        model_eval = DeepSADModelEval(optimizer_, eta)
        model_eval.set_network(net_name)
        model_eval.load_model(model_path=model_path, map_location=device)

        # Test the model
        model_eval.test(dataset_eval,
                        eta=eta,
                        batch_size=batch_size,
                        device=device,
                        n_jobs_dataloader=n_jobs_dataloader)
        _, _, scores = zip(*model_eval.results['test_scores'])
        y = [1 if e > cut else 0 for e in scores]
        recall = sum(y) / len(y)
        total_recall.append(recall)

        # Save the results
        f.write('--------------------\n')
        f.write('[Recall for file {}] {}\n'.format(i, recall))
        print('[Recall for file {}] {}\n'.format(i, recall))

    total_recall = np.array(total_recall)
    mean_recall = total_recall.mean()
    std_recall = total_recall.std()
    f.write('\n[**Recall Mean**] {}\n\n[**Recall std**] {}\n'.format(mean_recall, std_recall))
    print('\n[**Recall Mean**] {}\n[**Recall std**] {}\n'.format(mean_recall, std_recall))

f.write('###########################################################\n\n\n\n')
f.close()
print('Finished. Now I am going to bed. Bye.')
