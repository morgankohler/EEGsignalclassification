import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


# class ResidualLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, n_layers, bidirectional, dropout):
#         super(ResidualLSTM, self).__init__()
#
#         lstms = []
#         for i in range(n_layers):
#             if i == 0:
#                 lstm_input_dim = input_dim
#             else:
#                 lstm_input_dim = hidden_dim
#             lstms.append(
#                 nn.LSTM(lstm_input_dim,
#                         hidden_dim,
#                         num_layers=1,
#                         bidirectional=False,
#                         dropout=0)
#             )
#         self.lstm_forward = nn.ModuleList(lstms)
#         self.lstm_backward = copy.deepcopy(self.lstm_forward)
#
#         self.residual_linear_projection_forward = nn.Linear(input_dim, hidden_dim)
#         self.residual_linear_projection_backward = nn.Linear(input_dim, hidden_dim)
#
#         self.do = nn.Dropout(dropout)
#
#     def forward(self, x):
#         output_forward, hidden_forward, cell_forward = self.process_direction(x,
#                                                                               self.lstm_forward,
#                                                                               self.residual_linear_projection_forward)
#         output_backward, hidden_backward, cell_backward = self.process_direction(x.flip(dims=[2]),
#                                                                                  self.lstm_forward,
#                                                                                  self.residual_linear_projection_backward)
#
#         final_out = torch.cat((output_forward, output_backward), dim=2)
#         final_hidden = torch.cat((hidden_forward, hidden_backward), dim=0)
#         final_cell = torch.cat((cell_forward, cell_backward), dim=0)
#
#         return final_out, (final_hidden, final_cell)
#
#     def process_direction(self, z, lstms, residual_projection):
#         output, hidden, cell = (0, 0, 0)
#         for idx, lstm in enumerate(lstms):
#             output, (hidden, cell) = lstm(z)
#             if idx == 0:
#                 z = residual_projection(z)
#             output = output + z
#             z = self.do(output)
#         return output, hidden, cell


class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, bidirectional, dropout):
        super(ResidualLSTM, self).__init__()

        self.n_layers = n_layers

        lstms = []
        for i in range(n_layers):
            if i == 0:
                lstm_input_dim = input_dim
            elif bidirectional:
                lstm_input_dim = hidden_dim * 2
            else:
                lstm_input_dim = hidden_dim
            lstms.append(
                nn.LSTM(lstm_input_dim,
                        hidden_dim,
                        num_layers=1,
                        bidirectional=bidirectional,
                        dropout=0)
            )
        self.lstms = nn.ModuleList(lstms)

        if bidirectional:
            self.residual_linear_projection = nn.Linear(input_dim, hidden_dim * 2)
        else:
            self.residual_linear_projection = nn.Linear(input_dim, hidden_dim)

        self.do = nn.Dropout(dropout)

        self.bns = nn.ModuleList([nn.BatchNorm1d(200)] * n_layers)

    def forward(self, x):
        output, hidden, cell = (0, 0, 0)
        for idx in range(self.n_layers):
            output, (hidden, cell) = self.lstms[idx](x)
            if idx == 0:
                x = self.residual_linear_projection(x)
            x = self.do(output + x)
            x = self.bns[idx](x.transpose(0,1)).transpose(0,1)

        return output, (hidden, cell)


class EEGCLassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, seq_length):
        super(EEGCLassifier, self).__init__()

        self.lstm = ResidualLSTM(input_dim,
                                 hidden_dim,
                                 n_layers,
                                 bidirectional,
                                 dropout)

        # self.lstm = nn.LSTM(input_dim,
        #                     hidden_dim,
        #                     num_layers=n_layers,
        #                     bidirectional=bidirectional,
        #                     dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        z = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        z = self.fc(z)

        return z
