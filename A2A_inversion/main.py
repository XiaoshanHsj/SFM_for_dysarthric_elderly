import sys,os,argparse
import torch
import numpy as np
sys.path.append("..")

parser = argparse.ArgumentParser(description = "Pytorch implementation of Autoencoder Model")
parser.add_argument('--cuda-id', type = int, choices = [0, 1, 2], help = 'set cuda device id')
parser.add_argument('--seed', type = int, default = 88, help = 'set initialization seed, default: 88')
parser.add_argument('--train', action = 'store_true', help = 'train mode')
parser.add_argument('--test', action = 'store_true', help = 'test mode')
parser.add_argument('--plot-loss', action = 'store_true', help = 'use the train.log to draw a plot of the loss changing trend')
parser.add_argument('--n-input', type = int, default = 768, help = 'dimension of input layer')
# parser.add_argument('--n-dnn-layers', type = int, default = 2, help = 'number of linear layers, default = 2')
# parser.add_argument('--n-dnn-hiddens', type = int, default = 512, help = 'number of hidden nodes per linear layer, default = 512')
parser.add_argument('--n-arti-output', type = int, default = 144, help = 'dimension of output articulatory')
# parser.add_argument('--n-phone-output', type = int, default = 40, help = 'dimension of output phone')
parser.add_argument('--n-epochs', type = int, default = 40, help = 'maximum number of epochs for training, default: 100')
parser.add_argument('--epoch-report', type = int, default = 8, help = 'report model for every epoch-report epochs, default: 10')
parser.add_argument('--lr', type = float, default = 0.0005, help = 'initial learning rate, default: 0.0001')
parser.add_argument('--train-load-num', type = int, default = 32, help = 'train file number to load each time, default: 512')
parser.add_argument('--valid-load-num', type = int, default = 32, help = 'valid file number to load each time, default: 512')
args = parser.parse_args()

gpu_id = args.cuda_id
seed = args.seed
epoch_num = args.n_epochs
epoch_report = args.epoch_report
layer_sizes_no_out = [args.n_input] + [512, 256, 128, 64]
out_arti_size = args.n_arti_output
# out_phone_size = args.n_phone_output
# n_dnn_layers = args.n_dnn_layers
lr = args.lr
train_load_num = args.train_load_num
valid_load_num = args.valid_load_num
torch.backends.cudnn.benchmark=True

if torch.cuda.is_available() and gpu_id >= 0:
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
else:
    torch.manual_seed(seed)
np.random.seed(seed)

from DNN_based_model import *
from DNN_based_model.inversion_model import A2A_inversion_model

# draw train and valid regression loss changing plot
train_log = 'train.log'

train_input_path = 'train.scp'
valid_input_path = 'valid.scp'

unseen_test_input_path = 'unseen_test.scp'
seen_test_input_path = 'seen_test.scp'

# get_inverted_path = 'lib/flists/inverted/dbank_evde.scp'
get_inverted_path = 'test.scp'

out_ark = 'invertied_dct_output/tal_test.ark'
out_scp = 'invertied_dct_output/tal_test.scp'

resume_model_path = None # use a absolute model path here to resume training if necessary
test_model_path = None   # use a user-defined absolute model path if necessary

model = A2A_inversion_model(layer_sizes_no_out ,out_arti_size, train_input_path, valid_input_path, unseen_test_input_path, seen_test_input_path, \
        gpu_id, lr, train_load_num, valid_load_num, mdn_gaussian_num = 1)
if args.train:
    model.train(epoch_num, resume_model_path, epoch_report)
elif args.test:
    model.test(get_inverted_path, out_ark, out_scp, test_model_path)

