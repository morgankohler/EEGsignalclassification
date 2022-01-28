import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import time
import os
import argparse
from pathlib import Path

from dummy_loader import DummyLoader
from cla_dataloader import CLADataset
from model import EEGCLassifier
from utils import get_accuracy


def train(args):
    torch.manual_seed(args.seed)

    writer = SummaryWriter(log_dir=args.log_save_dir)

    if args.use_dummy_loader:
        train_set = DummyLoader(root=args.data_root, train=True)
        valid_set = DummyLoader(root=args.data_root, train=False)
    else:
        train_set = CLADataset(root=args.data_root, train=True)
        valid_set = CLADataset(root=args.data_root, train=False)

    train_queue = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)
    valid_queue = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size)

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
    model = model.cuda()

    ce_loss = nn.CrossEntropyLoss()

    print(f'Training from epoch: {start_epoch} ...')
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs+1):
        model.train()

        epoch_time = time.time()
        loss_vals = []
        for x, y in train_queue:
            optimizer.zero_grad()
            y_hat = model(x.cuda())
            loss = ce_loss(y_hat, y.cuda())
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
        f = open(args.train_output_file, 'w')
        f.write(train_out)
        f.close()

        if epoch % args.save_every == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        }, os.path.join(f'{args.model_save_dir}', f'{epoch}.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_number', type=int, default=7,
                        help='Train number to track experiments.')
    parser.add_argument('--from_checkpoint', type=bool, default=False,
                        help='Whether start training from checkpoint.')
    parser.add_argument('--checkpoint', type=int, default=100,
                        help='Checkpoint epoch to load.')
    parser.add_argument('--data_root', type=str, default='../data/CLA-3states/parsed/',
                        help='Data save root dir.')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save every n epochs.')
    parser.add_argument('--use_dummy_loader', type=bool, default=False,
                        help='Whether to use real data or dummy data for testing purposes.')

    # general hyperparameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='Max learning rate for cosine annealing lr schedule.')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='Min learning rate for cosine annealing lr schedule.')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for random initializations.')

    # dataset specific params
    parser.add_argument('--seq_length', type=int, default=200,
                        help='Length of eeg seq data for model input (depends on how data is processed)')
    parser.add_argument('--input_dim', type=int, default=21,
                        help='Dimension of eeg data (depends on dataset)')

    # model specific params
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dim for LSTM and attention blocks.')
    parser.add_argument('--output_dim', type=int, default=3,
                        help='Output vector dimension.')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Numer of lstm-attention layers.')
    parser.add_argument('--bidirectional', type=bool, default=True,
                        help='Whether lstm is bidirectional or not.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout in attention fc layers.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads in the attention blocks.')
    parser.add_argument('--attn_dim', type=int, default=32,
                        help='Dimension of q,k,v attention vectors within each head.')

    args = parser.parse_args()
    print(args)

    args.data_root = Path(args.data_root)
    args.checkpoint = os.path.join(f'models/{args.train_number}', f'{args.checkpoint}.pth')
    args.model_save_dir = os.path.join(f'models', f'{args.train_number}')
    args.log_save_dir = os.path.join(f'logs', f'{args.train_number}')
    args.train_output_file = os.path.join(f'train_out', f'{args.train_number}.out')
    if not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.isdir(args.log_save_dir):
        os.makedirs(args.log_save_dir)

    train(args)
