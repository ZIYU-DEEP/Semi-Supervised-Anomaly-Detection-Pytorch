"""
Title: folder_to_xs.py
Description: Preprocessing for spectrum data.
Author: Lek'Sai Ye, University of Chicago
"""

import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='forecast', help='[Choice] forecast, deepsad')
parser.add_argument('--normal_folder', type=str, default='ryerson')
parser.add_argument('--abnormal_folder', type=str, default='downtown_LOS-5M-USRP1')
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--predict_size', type=int, default=25)
parser.add_argument('--n_features', type=int, default=128) 
