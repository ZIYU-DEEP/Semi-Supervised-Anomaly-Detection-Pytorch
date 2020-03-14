"""
Title: main.py
Description: The main file to run the unsupervised models.
Author: Leksai Ye, University of Chicago
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
import argparse
import numpy as np
import pandas as pd
from main_loading import *
from main_network import *
from main_model import *

# Most of the time, you only need to specify:
# normal_filename, abnormal_filename, optimizer_, eta_str, n_epochs

# Arguments for main_loading
parser = argparse.ArgumentParser()
parser.add_argument('--loader_name', type=str, default='forecast_unsupervised',
                    help='[Choise]: forecast, forecast_unsupervised')
parser.add_argument('--loader_eval_name', type=str, default='forecast_eval',
                    help='[Choise]: forecast_eval')
parser.add_argument('--root', type=str, default='/net/adv_spectrum/array_data')
parser.add_argument('--normal_file', type=str, default='downtown_big_normal')
parser.add_argument('--abnormal_file', type=str, default='_',
                    help='[Example]: _, downtown_sigOver_10ms_big_abnormal')
parser.add_argument('--random_state', type=int, default=42)
parser.add_argument('--in_size', type=int, default=100)
parser.add_argument('--out_size', type=int, default=25)
parser.add_argument('--n_features', type=int, default=128)
parser.add_argument('--train_portion', type=float, default=0.8)

# Arguments for main_network
parser.add_argument('--net_name', type=str, default='gru',
                    help='[Choice]: gru, gru_stacked')

# Arguments for main_model
parser.add_argument('--optimizer_', type=str, default='forecast_unsupervised',
                    help='[Choice]: forecast_unsupervised, forecast_exp, forecast_minus')
parser.add_argument('--eta_str', default=100,
                    help='The _% represenntation of eta - choose from 100, 50, 25, etc.')
parser.add_argument('--optimizer_name', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--lr_milestones', type=tuple, default=(80, 120))
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--device_no', type=int, default=1)
parser.add_argument('--n_jobs_dataloader', type=int, default=0)
parser.add_argument('--save_ae', type=bool, default=True,
                    help='Only apply to Deep SAD model.')
parser.add_argument('--load_ae', type=bool, default=False,
                    help='Only apply to Deep SAD model.')
parser.add_argument('--fp_rate', type=float, default=0.05,
                    help='The false positive rate as the judge threshold.')

# Arguments for output_paths
parser.add_argument('--test_list_filename', type=str, default='downtown_test_list.npy')
parser.add_argument('--txt_filename', type=str, default='full_results.txt')
p = parser.parse_args()

# Extract the arguments
loader_name = p.loader_name
root, normal_file, abnormal_file = p.root, p.normal_file, p.abnormal_file
random_state, in_size, out_size = p.random_state, p.in_size, p.out_size
n_features, train_portion, net_name = p.n_features, p.train_portion, p.net_name
optimizer_, eta_str, optimizer_name = p.optimizer_, p.eta_str, p.optimizer_name
lr, n_epochs, lr_milestones, batch_size = p.lr, p.n_epochs, p.lr_milestones, p.batch_size
weight_decay, device_no, n_jobs_dataloader = p.weight_decay, p.device_no, p.n_jobs_dataloader
save_ae, load_ae, fp_rate = p.save_ae, p.load_ae, p.fp_rate
test_list_filename, txt_filename = p.test_list_filename, p.txt_filename

# Define data filenames
normal_filename = '{}.npy'.format(normal_file)
abnormal_filename = '{}.npy'.format(abnormal_file)

# Define folder names
folder_name = '{}_{}_{}'.format(normal_file, abnormal_file, optimizer_)
out_path = './{}'.format(folder_name)

# Define the general txt file path with stores all models' results
txt_result_file = './{}'.format(txt_filename)

# Define the resulting file paths
file_str = '{}_{}_{}_{}'.format(folder_name, net_name, n_epochs, eta_str)
model_path = '{}/model_{}.tar'.format(out_path, file_str)
results_path = '{}/results_{}.json'.format(out_path, file_str)
result_df_path = '{}/result_df_{}.pkl'.format(out_path, file_str)
cut_path = '{}/cut_{}.pkl'.format(out_path, file_str)

# Define additional stuffs
device = 'cuda:{}'.format(device_no)
eta = float(eta_str * 0.01)
test_list = np.load('../utils/{}'.format(test_list_filename))

# Check the existence of output path
if not os.path.exists(out_path):
    os.makedirs(out_path)

#############################################
# 1. Model Training
#############################################
# Loading data
dataset = load_dataset(loader_name, root, normal_filename, abnormal_filename,
                       random_state, in_size, out_size, n_features, train_portion)

# Loading model
model = Model(optimizer_, eta)

# Training model
model.set_network(net_name)
model.train(dataset, eta, optimizer_name, lr, n_epochs, lr_milestones,
            batch_size, weight_decay, device, n_jobs_dataloader)


#############################################
# 2. Model Testing
#############################################
# Test and Save model
model.test(dataset, device, n_jobs_dataloader)
model.save_results(export_json=results_path)
model.save_model(export_model= model_path, save_ae=save_ae)

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
f.write('=====================\n')
f.write('[DataFrame Name] {}\n'.format(result_df_path))
f.write('[Normal Filename] {}\n'.format(normal_filename))
f.write('[Abnormal Filename] {}\n'.format(abnormal_filename))
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
for test_abnormal_filename in test_list:
    # Load dataset for evaluation
    dataset_eval = load_dataset(loader_eval_name, root, normal_filename, abnormal_filename,
                                random_state, in_size, out_size, n_features, train_portion)
    # Load model for evaluation
    model_eval = ModelEval(optimizer_, eta=eta)
    model_eval.set_network(net_name)
    model_eval.load_model(model_path=model_path, map_location=device)

    # Test the model
    model_eval.test(dataset_eval, device=device, eta=eta)
    _, _, scores = zip(*model_eval.results['test_scores'])
    _, _, scores = np.array(indices), np.array(labels), np.array(scores)
    y = [1 if e > cut else 0 for e in scores]

    # Save the results
    f.write('---------------------\n')
    f.write('[Recall for {}] {}\n'.format(test_abnormal_filename, sum(y) / len(y)))
    print('Detection result for the file: {}'.format(test_abnormal_filename))
    print(sum(y) / len(y))

f.write('=====================\n\n')
f.close()
print('Finished. Now I am going to bed. Bye.')
