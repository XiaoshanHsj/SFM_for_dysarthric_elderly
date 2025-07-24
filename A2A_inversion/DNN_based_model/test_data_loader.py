import torch
import torch.utils.data as data
import numpy as np
import os, sys
import kaldiio

class Autoencoder_Dataset(data.Dataset):
    def __init__(self, path_list):
        self.feature_list = []
        with open(path_list) as f_r:
            for line in f_r:
                line_strip = line.strip()
                terms = line_strip.split()
                idx, path = terms[0], terms[1]
                self.feature_list.append((idx, path))

    def __getitem__(self, index):
        feature_path = self.feature_list[index][1]

        input_data = kaldiio.load_mat(feature_path)[:, 0: 256]
        row_num, col_num = input_data.shape
        X_input = np.zeros((row_num, 3 * col_num))
        for i in range(row_num):
            if i == 0:
                X_input[i, :] = np.hstack((input_data[i, :],input_data[i, :],input_data[i + 1, :]))
            elif i == row_num - 1:
                X_input[i, :] = np.hstack((input_data[i - 1, :],input_data[i, :],input_data[i, :]))
            else:
                X_input[i, :] = np.hstack((input_data[i - 1, :],input_data[i, :],input_data[i + 1, :]))
        return X_input

    def __len__(self):
        return len(self.feature_list)

def thread_testloaddata(path_list, file_num):
    path_list = os.path.abspath(path_list)
    dataset = Autoencoder_Dataset(path_list)
    loader = data.DataLoader(
        dataset = dataset,
        batch_size = file_num,
        shuffle = False,
        num_workers = 3,
        collate_fn = test_collate_data,
    )
    return loader

def test_collate_data(batch):
    input_data = list(zip(*batch))
    test_input = torch.FloatTensor(input_data)
    return test_input

