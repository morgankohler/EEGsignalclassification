import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
# from tensorboardX import SummaryWriter

import time
import os

from cla_dataloader import CLADataset
from model import LSTM
from utils import get_accuracy


class Params:
    pass
args = Params()

train_number = 1

args.out_folder = f'models/{train_number}'
args.log_dir = f'logs/{train_number}'
try:
    os.mkdir(args.out_folder)
    os.mkdir(args.log_dir)
except:
    pass

args.save_every = 5

# training params
args.batch_size = 16
args.learning_rate = 0.01
args.learning_rate_min = 0.0001
args.epochs = 100
args.start_epoch = 1

# model params
args.input_dim = 21
args.hidden_dim = 128
args.output_dim = 3
args.n_layers = 1
args.bidirectional = True
args.dropout = 0.5

# writer = SummaryWriter(log_dir=args.log_dir)

train_set = CLADataset(train=True)
train_queue = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)

valid_set = CLADataset(train=False)
valid_queue = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size)

model = LSTM(args.input_dim, args.hidden_dim, args.output_dim, args.n_layers, args.bidirectional, args.dropout)

optimizer = optim.Adamax(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs + 1, eta_min=args.learning_rate_min)

ce_loss = nn.CrossEntropyLoss()

print(f'Training from epoch: {args.start_epoch} ...')
start_time = time.time()
for epoch in range(args.start_epoch, args.epochs+1):
    model.train()

    epoch_time = time.time()
    loss_vals = []
    for x, y in train_queue:
        x = x.transpose(0,1) # .cuda()

        optimizer.zero_grad()
        y_hat = model(x)
        loss = ce_loss(y_hat, y) # y.cuda()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss)
    scheduler.step()

    train_accuracy = get_accuracy(model, train_queue, args.batch_size)
    valid_accuracy = get_accuracy(model, valid_queue, args.batch_size)

    loss = sum(loss_vals)/len(loss_vals)

    # writer.add_scalar('loss', loss, epoch)
    # writer.add_scalar('train acc', train_accuracy, epoch)
    # writer.add_scalar('valid acc', valid_accuracy, epoch)

    end_time = time.time()
    print(f'epoch: {epoch},'
          f'program runtime: {(end_time - start_time):.1f},'
          f'epoch runtime: {(end_time - epoch_time):.1f},'
          f'training loss: {loss:.5f},'
          f'training accuracy: {train_accuracy:.5f}',
          f'valid accuracy: {valid_accuracy:.5f}')

    if epoch % args.save_every == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    }, f'{args.out_folder}/{epoch}.pth')

    scheduler.step()