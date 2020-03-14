"""
Title: utils.py
Description: Helper functions.
Author: Leksai Ye, University of Chicago
"""

import numpy as np


#############################################
# 0. Split data to windows
#############################################
def window_data(array, in_size, out_size, overlap=False, n_features=128):
    """
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
        array = array.reshape(-1, n_features)

    len_ = len(array)
    win_size = in_size + out_size

    if overlap:
        start_index = [i for i in range(0, len_, out_size)]
    else:
        start_index = [i for i in range(0, len_, win_size)]
    if start_index[-1] + win_size > len_: start_index.pop()

    index_in = [list(range(i, i + in_size)) for i in start_index]
    index_out = [list(range(i + in_size, i + in_size + out_size)) for i in start_index]

    array_in = array[index_in, :]
    array_out = array[index_out, :]

    return array_in, array_out
