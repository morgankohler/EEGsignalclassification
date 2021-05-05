import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.lstm = nn.LSTM(input_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        z = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        z = self.fc(z)

        return z
