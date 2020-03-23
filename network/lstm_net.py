"""
Title: gru_net.py
Description: The LSTM network.
Author: Lek'Sai Ye, University of Chicago
"""

import torch.nn as nn
from base_net import BaseNet


class LSTMNet(BaseNet):
    # Do not use this one.
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(128, 64, num_layers=3, batch_first=True)
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


class LSTMNetStacked(BaseNet):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(128, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 128 * 25)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, (h, c) = self.lstm3(x)
        x = self.fc(h.view(-1, h.shape[-1]))
        # x = self.act(x)
        x = x.view(-1, 25, 128)
        return x
