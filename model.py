import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class AttentionHead(nn.Module):
    def __init__(self, attention_dim, model_dim):
        super(AttentionHead, self).__init__()

        self.model_dim = model_dim

        self.Q = nn.Linear(model_dim, attention_dim, bias=False)
        self.K = nn.Linear(model_dim, attention_dim, bias=False)
        self.V = nn.Linear(model_dim, attention_dim, bias=False)

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        z = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.model_dim)
        z = F.softmax(z, dim=2)
        z = torch.bmm(z, v)
        return z


class TFEncoder(nn.Module):
    def __init__(self, args):
        super(TFEncoder, self).__init__()

        self.num_heads = args.num_heads
        self.attention_dim = args.attention_dim

        attention_heads = []
        for idx_heads in range(args.num_heads):
            attention_heads.append(AttentionHead(args.attention_dim, args.model_dim))
        self.attention_heads = nn.ModuleList(attention_heads)

        self.multi_head_weight = nn.Linear(args.attention_dim * args.num_heads, args.model_dim, bias=False)

        self.layer_norm_1 = nn.LayerNorm(args.model_dim)
        self.layer_norm_2 = nn.LayerNorm(args.model_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(args.model_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.model_dim)
        )

        self.drop_out = nn.Dropout(args.drop_out)

    def forward(self, z):
        z_concat_heads = torch.tensor([]).cuda()
        for i, head in enumerate(self.attention_heads):
            z_concat_heads = torch.cat((z_concat_heads, head(z)), 2)

        z_concat_heads = self.multi_head_weight(z_concat_heads)
        z_concat_heads = self.drop_out(z_concat_heads)

        z = z_concat_heads + z

        z = self.layer_norm_1(z)

        ff = self.drop_out(self.feed_forward(z))

        z = ff + z

        z = self.layer_norm_2(z)

        return z


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.linear_in = nn.Linear(21, 256)
        self.linear_out = nn.Linear(256, args.output_dim)

        self.pos_embedding = nn.Embedding(embedding_dim=args.model_dim, num_embeddings=args.max_seq_length)

        encoder_blocks = []
        for idx_blocks in range(args.num_encoder_blocks):
            encoder_blocks.append(TFEncoder(args))

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        self.drop_out = nn.Dropout(args.drop_out)

    def forward(self, x):

        x = self.linear_in(x)

        b, t, e = x.size()

        positions = self.pos_embedding(torch.arange(t).cuda())[None, :, :].expand(b, t, e)
        # positions = self.pos_embedding(torch.arange(t))[None, :, :].expand(b, t, e)
        x = x + positions

        x = self.drop_out(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = x.mean(dim=1)
        x = self.linear_out(x)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, attn_dim, model_dim):
        super(MultiHeadSelfAttention, self).__init__()

        self.attention_heads = nn.ModuleList([AttentionHead(attn_dim, model_dim)] * num_heads)

        self.multi_head_weight = nn.Linear(attn_dim * num_heads, model_dim, bias=False)

    def forward(self, z):
        z_concat_heads = torch.tensor([]).cuda()
        for i, head in enumerate(self.attention_heads):
            z_concat_heads = torch.cat((z_concat_heads, head(z)), 2)

        return self.multi_head_weight(z_concat_heads)


class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, bidirectional, dropout, num_heads, attn_dim):
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
            self.attention_layers = nn.ModuleList([MultiHeadSelfAttention(num_heads, attn_dim, hidden_dim * 2)] * n_layers)
            self.lns = nn.ModuleList([nn.LayerNorm(hidden_dim * 2)] * (n_layers*2))
        else:
            self.residual_linear_projection = nn.Linear(input_dim, hidden_dim)
            self.attention_layers = nn.ModuleList([MultiHeadSelfAttention(num_heads, attn_dim, hidden_dim)] * n_layers)
            self.lns = nn.ModuleList([nn.LayerNorm(hidden_dim)] * (n_layers*2))

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        z, hidden, cell = (0, 0, 0)
        for idx in range(self.n_layers):
            z, (hidden, cell) = self.lstms[idx](x)
            if idx == 0:
                x = self.residual_linear_projection(x)
            z = self.do(z) + x

            # transpose from (seq_length, batch, model_dim) to (batch, seq_length, model_dim) for attn portion
            z = z.transpose(0, 1)

            z = self.lns[idx*2](z)
            z = self.do(self.attention_layers[idx](z)) + z
            z = self.lns[idx*2 + 1](z)

            z = z.transpose(0, 1)

        z = torch.mean(z, dim=0)

        return z


class EEGCLassifier(nn.Module):
    def __init__(self, args):
        super(EEGCLassifier, self).__init__()

        self.lstm = ResidualLSTM(args.input_dim,
                                 args.hidden_dim,
                                 args.n_layers,
                                 args.bidirectional,
                                 args.dropout,
                                 args.num_heads,
                                 args.attn_dim)

        # self.lstm = nn.LSTM(input_dim,
        #                     hidden_dim,
        #                     num_layers=n_layers,
        #                     bidirectional=bidirectional,
        #                     dropout=dropout)

        self.fc = nn.Linear(args.hidden_dim * 2, args.output_dim)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):

        z = self.lstm(x)

        # z = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        z = self.dropout(z)

        z = self.fc(z)

        return z
