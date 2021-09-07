import torch

from model import EEGCLassifier

class params:
    pass

args = params()

args.batch_size = 256
args.learning_rate = 0.01
args.learning_rate_min = 0.0001
args.epochs = 200

# model params
args.seq_length = 200
args.input_dim = 21
args.hidden_dim = 128
args.output_dim = 3
args.n_layers = 12
args.bidirectional = True
args.dropout = 0.5

data = torch.rand([200, 2, 21])

model = EEGCLassifier(args.input_dim, args.hidden_dim, args.output_dim, args.n_layers, args.bidirectional, args.dropout, args.seq_length)
model = model

_=model(data)
_=0