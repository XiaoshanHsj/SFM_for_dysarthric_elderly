import os, sys, random, time, re
import torch
import torch.nn as nn
from . import mdn

class Network(nn.Module):
    def __init__(self, layer_sizes_no_out, out_arti_size, mdn_gaussian_num = 1):
        super(Network, self).__init__()
        self.layer_sizes_no_out = layer_sizes_no_out
        # articulatory
        self.dnn_layers = self.multi_linearlayer(layer_sizes_no_out)

        self.linear_for_arti = nn.Sequential()
        self.linear_for_arti.add_module('linear_arti', nn.Linear(layer_sizes_no_out[-1], out_arti_size))

        self.mdn = mdn.MDN(out_arti_size, out_arti_size, mdn_gaussian_num)

    def multi_linearlayer(self, layer_sizes_no_out):
        dnn_num_layers = len(layer_sizes_no_out) - 1
        dnn_layers = nn.Sequential()
        in_layers = layer_sizes_no_out[0 : dnn_num_layers]
        out_layers = layer_sizes_no_out[1 : dnn_num_layers + 1]
        for l, (n_in, n_out) in enumerate(zip(in_layers, out_layers)):
            dnn_layers.add_module('linear' + str(l), nn.Linear(n_in, n_out))
            dnn_layers.add_module('relu' + str(l), nn.ReLU())
            # dnn_layers.add_module('dropout' + str(l), nn.Dropout(0.2))
        return dnn_layers

    def forward(self, x):
        linear_output = self.dnn_layers(x)
        arti_mid_output = self.linear_for_arti(linear_output)
        output_arti = self.mdn(arti_mid_output)
        return output_arti

'''
########### TEST ###########
import numpy as np
layer_sizes = [120, 256, 128, 64, 128]
dnn = Network(layer_sizes, 144)
print(dnn)
b_x = np.random.rand(1024, 120).astype(np.float32)
b_x = torch.FloatTensor(b_x)
output = dnn(b_x)
print(output.shape)
############ TEST ###########
'''