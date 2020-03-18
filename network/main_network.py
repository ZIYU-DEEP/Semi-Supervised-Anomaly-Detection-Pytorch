"""
Title: main_network.py
Description: Build networks.
Author: Lek'Sai Ye, University of Chicago
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/networks
"""

from lstm_net import LSTMNet, LSTMNetStacked


def build_network(net_name='lstm'):
    known_networks = ('lstm', 'lstm_stacked')
    assert net_name in known_networks

    net_name = net_name.strip()

    if net_name == 'lstm':
        return LSTMNet()

    if net_name == 'lstm_stacked':
        return LSTMNetStacked()

    return None
