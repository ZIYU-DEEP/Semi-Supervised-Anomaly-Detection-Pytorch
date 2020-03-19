"""
Title: utils.py
Description: Helper functions for loading data.
Author: Lek'Sai Ye, University of Chicago
"""

import os
import glob
import numpy as np
from pathlib import Path

# ############################################
# 0. Load Data from TXT File
# ############################################
def file_to_array(filename, n_channels=128):
    """
    This is a helper function which will be used in the subsequent function.
    Inputs:
        filename (str): e.g. '/net/adv_spectrum/data/feature/downsample_10/normal/downtown/100_25/*.txt'
        n_channels (int): the number of features
    Returns:
        array (np.array): shape --> (n_instances, n_channels)
    """
    print(filename)

    array = []
    with open(filename, 'r') as f:
        for line in f:
            # Each line should have 16000 = 125 * 128 values
            x = list(map(np.float32, line.split()))

            if len(x) % n_channels:
                # The following will be executed
                # If len(x) % n_channels != 0
                print('Incomplete data. Kill the creater!')
                return False

            array.extend(x)

    array = np.array(array).reshape((-1, n_channels))
    return array


# ############################################
# 1. Split data to windows For Forecast Model
# ############################################
def window_data(array, in_size, out_size, overlap=False, n_features=128):
    """
    This is a helper function which will be used in the subsequent function.
    Input:
        array (np.array): shape is (n_instances, n_features)
        in_size (int): the size (no. of instances) for a window's prefix
        out_size (int): the size for a window's suffix
        n_features (int): the no. of features
    Returns:
        array_in (np.array): shape is (n_instances, in_size, n_features)
        array_out (np.array): shape is (n_instances, out_size, n_features)
    """
    if len(array.shape) > 2:
        array = array.reshape(-1, int(n_features))

    len_ = len(array)
    win_size = in_size + out_size

    if overlap:
        start_index = [i for i in range(0, len_, out_size)]
    else:
        start_index = [i for i in range(0, len_, win_size)]

    while start_index[-1] + win_size + out_size > len_:
        start_index.pop()

    index_in = [list(range(i, i + in_size)) for i in start_index]
    index_out = [list(range(i + in_size, i + in_size + out_size)) for i in start_index]
    print(index_out)

    array_in = array[index_in, :]
    array_out = array[index_out, :]

    return array_in, array_out


# ############################################
# 2. Process File with Regard to Training
# ############################################
# ============================================
# 2. (a) For Forecast Model
# ============================================
def file_to_Xs(filename, in_size=100, out_size=25, overlap=True, n_features=128, train_portion=0.8, train=1):
    """
    This is a helper function which will be used in the subsequent function.
    Inputs:
        filename (str): e.g. '/net/adv_spectrum/data/feature/downsample_10/normal/downtown/100_25/*.txt'
        train (int): train = 0 --> test; train = 1 --> train; train = - 1 --> do not split
        overlap (bool): train = 1 / 0 --> True; train = - 1 --> False
    Returns:
        X_in (np.array): The input which will be fed into the network
                         shape --> (n_instances, in_size, n_features)
        X_out (np.array): The array which will be compared with net(X_in)
                         shape --> (n_instances, out_size, n_features)
    """
    X = file_to_array(filename)
    print(X.shape)

    if train == 1:
        X = X[:int(len(X) * train_portion)]
    elif train == 0:
        X = X[int(len(X) * train_portion):]
    elif train == - 1:
        X = X

    X_in, X_out = window_data(X, in_size, out_size, overlap, n_features)

    return X_in, X_out


# ============================================
# 2. (a) For DeepSAD Model
# ============================================
def file_to_Xs_deepsad(filename, window_size, n_features=128, train_portion=0.8, train=1):
    """
    This is a helper function which will be used in the subsequent function.
    Inputs:
        filename (str): e.g. '/net/adv_spectrum/data/feature/downsample_10/normal/downtown/100_25/*.txt'
        train (int): train = 0 --> test; train = 1 --> train; train = - 1 --> do not split
        overlap (bool): train = 1 / 0 --> True; train = - 1 --> False
    Returns:
        X_in (np.array): The input which will be fed into the network
                         shape --> (n_instances, in_size, n_features)
        X_out (np.array): The array which will be compared with net(X_in)
                         shape --> (n_instances, out_size, n_features)
    """

    # Read in data
    X = file_to_array(filename, n_features)

    # Reshape data
    n_instances = X.shape[0] // window_size
    new_shape_0 = n_instances * window_size
    X = X[:new_shape_0, :].reshape(- 1, window_size, n_features)
    print(X.shape)

    if train == 1:
        X = X[:int(len(X) * train_portion)]
    elif train == 0:
        X = X[int(len(X) * train_portion):]
    elif train == - 1:
        X = X

    return X


# ############################################
# 3. Save the "Large" Array for Training
# ############################################
# ============================================
# 3. (a) For Forecast Model
# ============================================
def folder_to_Xs_save(path, path_, in_size=100, out_size=25, overlap=True, n_features=128, train_portion=0.8, train=1):
    """
    The purpose of this function is the save the "big" arrays.
    You should only use this function for normal data, or sigOver data. No other fbs data.

    Inputs:
        path (str): e.g. '/net/adv_spectrum/data/feature/downsample_10/abnormal/downtown_sigOver_10ms/100_25'
        path_ (str): e.g. '/net/adv_spectrum/torch_data/downtown/abnormal/downtown_sigOver_10ms'
        overlap (bool): train = 1 / 0 --> True; train = - 1 --> False
    Returns:
        X_in (np.array): The input which will be fed into the network
        X_out (np.array): The array which will be compared with net(X_in)
    """
    if not os.path.exists(path_):
        os.makedirs(path_)

    for i, filename in enumerate(sorted(glob.glob(path + '/*.txt'))):
        print(filename)
        X_in, X_out = file_to_Xs(filename, in_size, out_size, overlap, n_features, train_portion, train)
        if not i:
            print('I am processing the first file for you.')
            full_in, full_out = X_in, X_out
        else:
            print('Now I am concatenating the {}th file.'.format(i))
            full_in = np.concatenate((full_in, X_in))
            full_out = np.concatenate((full_out, X_out))

    if train == 1:
        np.save(Path(path_) / 'X_train_in.npy', full_in)
        np.save(Path(path_) / 'X_train_out.npy', full_out)
    elif train == 0:
        np.save(Path(path_) / 'X_test_in.npy', full_in)
        np.save(Path(path_) / 'X_test_out.npy', full_out)
    elif train == - 1:
        np.save(Path(path_) / 'X_in.npy', full_in)
        np.save(Path(path_) / 'X_out.npy', full_out)

    return True


# ============================================
# 3. (b) For DeepSAD Model
# ============================================
def folder_to_Xs_save_deepsad(path, path_, window_size=100, n_features=128, train_portion=0.8, train=1):
    """
    The purpose of this function is the save the "big" arrays.
    You should only use this function for normal data, or sigOver data. No other fbs data.

    Inputs:
        path (str): '/net/adv_spectrum/data/feature/downsample_10/abnormal/downtown_sigOver_10ms/100_25'
        path_ (str): '/net/adv_spectrum/torch_data_deepsad/100/downtown/abnormal/downtown_sigOver_10ms'
        overlap (bool): train = 1 / 0 --> True; train = - 1 --> False
    Returns:
        X_in (np.array): The input which will be fed into the network
        X_out (np.array): The array which will be compared with net(X_in)
    """
    if not os.path.exists(path_):
        os.makedirs(path_)

    print(path)
    for i, filename in enumerate(sorted(glob.glob(path + '/*.txt'))):
        X = file_to_Xs_deepsad(filename, window_size, n_features, train_portion, train)
        if not i:
            print('I am processing the first file for you.')
            full_X = X
        else:
            print('Now I am concatenating the {}th file.'.format(i))
            full_X = np.concatenate((full_X, X))

    if train == 1:
        np.save(Path(path_) / 'X_train.npy', full_X)
    elif train == 0:
        np.save(Path(path_) / 'X_test.npy', full_X)
    elif train == - 1:
        np.save(Path(path_) / 'X.npy', full_X)

    return True


# ############################################
# 4. Save the "Small" Arrays for Evaluating
# ############################################
# ============================================
# 4. (a) For Forecast Model
# ============================================
def save_abnormal_Xs(path, path_, in_size=100, out_size=25, overlap=False, n_features=128):
    """
    The purpose of this function is the save the "small" arrays in separate folders for anomaly.
    Each original txt file corresponds to a 'file_i' folder.
    That folder lives in the same directory with the previous "large" array, if any.

    Inputs:
        path (str): e.g. '/net/adv_spectrum/data/feature/downsample_10/abnormal/downtown_LOS-5M-USRP1/100_25'
        path_ (str): e.g. '/net/adv_spectrum/torch_data/downtown/abnormal/downtown_sigOver_10ms'
        overlap (bool): train = 1 / 0 --> True; train = - 1 --> False
    Returns:
        X_in (np.array): The input which will be fed into the network
        X_out (np.array): The array which will be compared with net(X_in)
    """
    for i, filename in enumerate(sorted(glob.glob(path + '/*.txt'))):
        X_in, X_out = file_to_Xs(filename, in_size, out_size, overlap, n_features, 0, - 1)
        path_out = Path(path_) / 'file_{}'.format(i)
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        print(path_out)
        np.save(path_out / 'X_in.npy', X_in)
        np.save(path_out / 'X_out.npy', X_out)
    print('All files saved. Got to go to bed.')

    return True


# ============================================
# 4. (b) For DeepSAD Model
# ============================================
def save_abnormal_Xs_deepsad(path, path_, window_size=100, n_features=128):
    """
    The purpose of this function is the save the "small" arrays in separate folders for anomaly.
    Each original txt file corresponds to a 'file_i' folder.
    That folder lives in the same directory with the previous "large" array, if any.

    Inputs:
        path (str): '/net/adv_spectrum/data/feature/downsample_10/abnormal/downtown_LOS-5M-USRP1/100_25'
        path_: '/net/adv_spectrum/torch_data_deepsad/100/downtown/abnormal/downtown_LOS-5M-USRP1'
    Returns:
        X_in (np.array): The input which will be fed into the network
        X_out (np.array): The array which will be compared with net(X_in)
    """
    for i, filename in enumerate(sorted(glob.glob(path + '/*.txt'))):
        # Read in data
        X = file_to_Xs_deepsad(filename, window_size, n_features, 0, - 1)

        # Make path
        path_out = Path(path_) / 'file_{}'.format(i)
        if not os.path.exists(path_out): os.makedirs(path_out)
        print(path_out)

        # Save data
        np.save(path_out / 'X.npy', X)
        print(X.shape)
    print('All files saved. Got to go to bed.')

    return True
