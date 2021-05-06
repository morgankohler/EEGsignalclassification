import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMRecurrentBlock(nn.Module):
    def __init__(self, hidden_dim, bidirectional, dropout, seq_length, final_layer=False, sub_layers=1):
        super(LSTMRecurrentBlock, self).__init__()

        self.final_layer = final_layer

        self.lstm = nn.LSTM(hidden_dim,
                           hidden_dim,
                           num_layers=sub_layers,
                           bidirectional=bidirectional,
                           dropout=0)

        # self.bn = nn.BatchNorm1d(seq_length)

        self.do = nn.Dropout(p=dropout)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x) + x
        z = hidden
        if not self.final_layer:
            # z = self.bn(x.transpose(0,1)).transpose(0,1)
            z = self.do(z)
        return z


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, seq_length):
        super(LSTM, self).__init__()

        self.pre_fc = nn.Linear(input_dim, hidden_dim)

        lstms = [LSTMRecurrentBlock(hidden_dim, bidirectional, dropout, seq_length, final_layer=False)] * (n_layers-1)
        lstms += [LSTMRecurrentBlock(hidden_dim, bidirectional, dropout, seq_length, final_layer=True)]
        self.lstm = nn.ModuleList(lstms)

        # self.lstm = nn.LSTM(input_dim,
        #                    hidden_dim,
        #                    num_layers=n_layers,
        #                    bidirectional=bidirectional,
        #                    dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        z = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        z = self.fc(z)

        return z
