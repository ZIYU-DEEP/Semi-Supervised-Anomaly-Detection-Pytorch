"""
Title: main_network.py
Description: Build networks.
Author: Leksai Ye, University of Chicago
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/networks
"""

from gru_net import GRUNet, GRUNetStacked


def build_network(net_name='lstm'):
    known_networks = ('lstm', 'lstm_stacked')
    assert net_name in known_networks

    net_name = net_name.strip()

    if net_name == 'lstm':
        return LSTMNet()

    if net_name == 'lstm_stacked':
        return LSTMNetStacked()

    return None
