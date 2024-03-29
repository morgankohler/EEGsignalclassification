import torch
from torch.utils import data
import numpy as np
import os


class CLADataset(data.Dataset):
    def __init__(self, root, train=True, split=0.8):
        super(CLADataset, self).__init__()

        if train:
            subjects = ['SubjectC', 'SubjectD', 'SubjectE', 'SubjectF']
        else:
            subjects = ['SubjectA', 'SubjectB']

        labels = torch.tensor([])
        data = torch.tensor([])
        for data_file in os.listdir(root):

            subject = data_file[data_file.find('Subject'):data_file.find('Subject')+8]
            if subject not in subjects:
                continue

            session_data = torch.load(os.path.join(root, data_file, 'data.pt'))
            data = torch.cat((data, session_data))
            session_labels = torch.load(os.path.join(root, data_file, 'labels.pt'))
            labels = torch.cat((labels, session_labels))

        # 22nd channel only informs on the stimuli activation (not important for stimulus response classification)
        data = data[:, :, :21]

        random_data_permutation = torch.randperm(data.shape[0])
        data = data[random_data_permutation]
        labels = labels[random_data_permutation]

        data = (data - data.mean()) / data.std()

        self.data = data
        # originally in range (1,3) need to subtract 1 to make it in range for cross entropy loss function
        self.labels = labels.long() - 1

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]


# d = CLADataset('/mnt/c/dev/eeg/data/CLA-3States/parsed', train=True)
# l = len(d)
# x = d[0]
# _=0
