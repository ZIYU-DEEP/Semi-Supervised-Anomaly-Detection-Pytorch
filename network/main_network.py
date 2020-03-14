"""
Title: main_network.py
Description: Build networks.
Author: Leksai Ye, University of Chicago
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/networks
"""

from .gru_net import GRUNet, GRUNetStacked


def build_network(net_name='gru'):
    known_networks = ('gru', 'gru_stacked')
    assert net_name in known_networks

    net_name = net_name.strip()

    if net_name == 'gru':
        return GRUNet()

    if net_name == 'gru_stacked':
        return GRUNetStacked()

    return None
