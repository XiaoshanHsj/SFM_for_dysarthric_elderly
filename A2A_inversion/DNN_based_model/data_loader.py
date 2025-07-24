import torch
import torch.utils.data as data
import numpy as np
import os, sys
import kaldiio
import time
 
class Inversion_Dataset(data.Dataset):
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
        output_data = kaldiio.load_mat(feature_path)[:, 296: 440]
        row_num, col_num = input_data.shape
        X_input = np.zeros((row_num, 3 * col_num))
        for i in range(row_num):
            if i == 0:
                X_input[i, :] = np.hstack((input_data[i, :],input_data[i, :],input_data[i + 1, :]))
            elif i == row_num - 1:
                X_input[i, :] = np.hstack((input_data[i - 1, :],input_data[i, :],input_data[i, :]))
            else:
                X_input[i, :] = np.hstack((input_data[i - 1, :],input_data[i, :],input_data[i + 1, :]))
        return X_input, output_data

    def __len__(self):
        return len(self.feature_list)

def thread_loaddata(path_list, file_num):
    path_list = os.path.abspath(path_list)
    dataset = Inversion_Dataset(path_list)
    loader = data.DataLoader(
        dataset = dataset,
        batch_size = file_num,
        shuffle = True,
        num_workers = 3,
        collate_fn = collate_data,
    )
    return loader

def collate_data(batch):
    input_list = []
    output_list = []
    for input_data, output_data in batch:
        input_list.append(input_data)
        output_list.append(output_data)
    input_numpy = np.concatenate(input_list, axis = 0)
    output_numpy = np.concatenate(output_list, axis = 0)
    # col = input_numpy.shape[1]
    # assert col == 768
    # concat_numpy = np.hstack((input_numpy, output_numpy))
    # np.random.shuffle(concat_numpy)
    train_input = torch.FloatTensor(input_numpy)
    train_output = torch.FloatTensor(output_numpy)
    assert train_input.size(1) == 768 and train_output.size(1) == 144
    return train_input, train_output