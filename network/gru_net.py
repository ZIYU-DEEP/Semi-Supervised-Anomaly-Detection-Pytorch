"""
Title: gru_net.py
Description: The GRU network.
Author: Leksai Ye, University of Chicago
"""

import torch.nn as nn
from .base_net import BaseNet


class GRUNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(128, 64, num_layers=1, batch_first=True)
        self.BN = nn.BatchNorm1d(100)
        self.fc = nn.Linear(64 * 100, 128 * 25)
        self.act = nn.ReLU()

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.BN(x)
        x = x.reshape(x.size(0), 64 * 100).contiguous()
        x = self.fc(x)
        x = x.reshape(-1, 25, 128).contiguous()
        x = self.act(x)
        return x


class GRUNetStacked(BaseNet):

    def __init__(self):
        super().__init__()

        self.gru1 = nn.GRU(128, 64, num_layers=1, batch_first=True)
        self.BN1 = nn.BatchNorm1d(100)
        self.gru2 = nn.GRU(64, 64, num_layers=1, batch_first=True)
        self.BN2 = nn.BatchNorm1d(100)
        self.gru3 = nn.GRU(64, 64, num_layers=1, batch_first=True)
        self.BN3 = nn.BatchNorm1d(100)
        self.fc = nn.Linear(64 * 100, 128 * 25)
        self.act = nn.ReLU()

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.BN1(x)
        x, _ = self.gru2(x)
        x = self.BN2(x)
        x, _ = self.gru3(x)
        x = self.BN3(x)
        x = x.reshape(x.size(0), 64 * 100).contiguous()
        x = self.fc(x)
        x = x.reshape(-1, 25, 128).contiguous()
        x = self.act(x)
        return x
