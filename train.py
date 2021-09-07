import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tensorboardX import SummaryWriter

import time
import os

from dummy_loader import DummyLoader
from cla_dataloader import CLADataset
from model import EEGCLassifier, Transformer
from utils import get_accuracy


class Params:
    pass
args = Params()

train_number = 7

args.out_folder = f'models/{train_number}'
args.log_dir = f'logs/{train_number}'
args.train_file = f'train/{train_number}.out'
if not os.path.isdir(args.out_folder):
    os.mkdir(args.out_folder)
if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)

args.from_checkpoint = False
args.checkpoint = 'models/4/150.pth'

args.data_root = '../data/CLA-3states/parsed/'

args.save_every = 5

# training params
args.batch_size = 256
args.learning_rate = 0.01
args.learning_rate_min = 0.0001
args.epochs = 300

# model params
args.seq_length = 200
args.input_dim = 21
args.hidden_dim = 128
args.output_dim = 3
args.n_layers = 6
args.bidirectional = True
args.dropout = 0.5
args.num_heads = 8
args.attn_dim = 32

# transformer args for tf test
# args.max_seq_length = 200
# args.model_dim = 256
# args.attention_dim = 32
# args.hidden_dim = 1024
# args.num_heads = 8
# args.num_encoder_blocks = 2
# args.drop_out = 0.5

print(vars(args))

writer = SummaryWriter(log_dir=args.log_dir)

train_set = CLADataset(root=args.data_root, train=True)
# train_set = DummyLoader(root=args.data_root, train=True)
train_queue = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)

valid_set = CLADataset(root=args.data_root, train=False)
# valid_set = DummyLoader(root=args.data_root, train=False)
valid_queue = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size)

# model = Transformer(args)
model = EEGCLassifier(args)
model = model.cuda()
optimizer = optim.Adamax(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs + 1, eta_min=args.learning_rate_min)

start_epoch = 1
if args.from_checkpoint:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] + 1
# model = model.cuda()

ce_loss = nn.CrossEntropyLoss()

print(f'Training from epoch: {start_epoch} ...')
start_time = time.time()
for epoch in range(start_epoch, args.epochs+1):
    model.train()

    epoch_time = time.time()
    loss_vals = []
    for x, y in train_queue:
        x = x.transpose(0,1).cuda()
        # x = x.cuda()

        optimizer.zero_grad()
        y_hat = model(x)
        loss = ce_loss(y_hat, y.cuda()) # y.cuda()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss)
    scheduler.step()

    train_accuracy = get_accuracy(model, train_queue, args.batch_size)
    valid_accuracy = get_accuracy(model, valid_queue, args.batch_size)

    loss = sum(loss_vals)/len(loss_vals)

    writer.add_scalar('loss', loss, epoch)
    writer.add_scalar('train acc', train_accuracy, epoch)
    writer.add_scalar('valid acc', valid_accuracy, epoch)

    end_time = time.time()
    train_out = f'epoch: {epoch}, '\
                f'program runtime: {(end_time - start_time):.1f}, '\
                f'epoch runtime: {(end_time - epoch_time):.1f}, '\
                f'training loss: {loss:.5f}, '\
                f'training accuracy: {train_accuracy:.5f}, '\
                f'valid accuracy: {valid_accuracy:.5f}'

    print(train_out)
    f = open(args.train_file, 'w')
    f.write(train_out)
    f.close()

    if epoch % args.save_every == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    }, f'{args.out_folder}/{epoch}.pth')

    scheduler.step()
