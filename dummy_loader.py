
import torch
from torch.utils import data
import numpy as np
import os

seed = 1234
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# occasionally useful wanting to test architectural change runs without errors
class DummyLoader(data.Dataset):
    def __init__(self, root, train=True, split=0.8):
        super(DummyLoader, self).__init__()

        self.data = torch.rand([1000, 200, 21])
        self.labels = torch.ones([1000]).long()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]
