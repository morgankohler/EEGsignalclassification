import scipy.io as io
import numpy as np
import torch
import os

from utils import islandinfo


def extract_session_data(data, markers):

    extracted_session_data = torch.tensor([])
    extracted_session_labels = torch.tensor([])

    for search_class_index in range(1, 7):
        class_islands = islandinfo(markers, trigger_val=search_class_index)
        class_data = torch.zeros(len(class_islands[0]), 200, 22)

        class_markers = torch.ones(len(class_islands[0])) * search_class_index

        for sample_index, sample_indices in enumerate(class_islands[0]):
            class_data[sample_index] = torch.from_numpy(data[sample_indices[0]:sample_indices[1]+1])[:200]

        extracted_session_data = torch.cat((extracted_session_data, class_data))
        extracted_session_labels = torch.cat((extracted_session_labels, class_markers))

    return extracted_session_data, extracted_session_labels


def loop_data_directory(data_dir, save_dir):

    for data_file in os.listdir(data_dir):

        if data_file == 'parsed':
            continue

        data_path = os.path.join(data_dir, data_file)

        save_path = os.path.join(save_dir, data_file)
        os.mkdir(save_path)

        session_data = io.loadmat(data_path)

        markers = session_data['o']['marker'][0][0].squeeze()
        data = session_data['o']['data'][0][0].squeeze()

        parsed_data, parsed_labels = extract_session_data(data, markers)

        torch.save(parsed_data, os.path.join(save_path, 'data.pt'))
        torch.save(parsed_labels, os.path.join(save_path, 'labels.pt'))


if __name__ == '__main__':

    eeg_data_dir = '/mnt/c/dev/eeg/data/CLA-3states/'

    parsed_data_save_dir = '/mnt/c/dev/eeg/data/CLA-3states/parsed/'

    loop_data_directory(eeg_data_dir, parsed_data_save_dir)
