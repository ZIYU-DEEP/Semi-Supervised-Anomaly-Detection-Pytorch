"""
Title: main_network.py
Description: Build networks.
Author: Lek'Sai Ye, University of Chicago
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/networks
"""

from lstm_net import LSTMNet, LSTMNetStacked
from lstm_ae_net import LSTMEncoder, LSTMDecoder, LSTMAutoencoder
from rec_net import RecEncoder, RecDecoder, RecAutoencoder


def build_network(net_name='lstm', rep_dim=10):
    known_networks = ('lstm', 'lstm_stacked', 'lstm_autoencoder', 'rec')
    assert net_name in known_networks

    net_name = net_name.strip()

    if net_name == 'lstm':
        return LSTMNet()

    if net_name == 'lstm_stacked':
        return LSTMNetStacked()

    if net_name == 'lstm_autoencoder':
        return LSTMEncoder(rep_dim=rep_dim)

    if net_name == 'rec':
        return RecAutoencoder(rep_dim=rep_dim)

    return None


def build_autoencoder(net_name='lstm_autoencoder', rep_dim=10):
    known_networks = ('lstm_autoencoder', 'rec')
    assert net_name in known_networks

    net_name = net_name.strip()

    if net_name == 'lstm_autoencoder':
        return LSTMAutoencoder(rep_dim=rep_dim)

    if net_name == 'rec':
        return RecAutoencoder(rep_dim=rep_dim)

    return None
