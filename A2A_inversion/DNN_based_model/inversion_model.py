import torch
import torch.nn as nn
import numpy as np
from .data_loader import thread_loaddata 
from .test_data_loader import thread_testloaddata
from .network import Network
import os,sys,time
from kaldiio import WriteHelper
from . import mdn
from .tools_learning import criterion_both

class A2A_inversion_model(object):
    def __init__(self, layer_sizes_no_out ,out_arti_size, train_path, valid_path, unseen_test_path, seen_test_path, \
            gpu_id, lr, train_load_num, valid_load_num, mdn_gaussian_num = 1):
        if torch.cuda.is_available() and gpu_id >= 0:
            torch.cuda.set_device(gpu_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.out_arti_size = out_arti_size
        self.train_loader = thread_loaddata(train_path, train_load_num)
        self.valid_loader = thread_loaddata(valid_path, valid_load_num)
        self.unseen_loader = thread_loaddata(unseen_test_path, valid_load_num)
        self.seen_loader = thread_loaddata(seen_test_path, valid_load_num)
        # self.phone_loss = nn.CrossEntropyLoss().to(self.device)
        self.network = Network(layer_sizes_no_out ,out_arti_size, mdn_gaussian_num).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.5, patience = 1)

    def train_loop(self):
        train_loss, train_num = 0.0, 0
        train_mdn_loss, train_pearson_loss, train_mse_loss = 0.0, 0.0, 0.0
        for step, (batch_x, batch_label) in enumerate(self.train_loader):
            b_x = batch_x.to(self.device)
            b_label = batch_label.to(self.device)
            train_num += b_x.size(0)
            train_output_arti = self.network(b_x)
            train_pi, train_sigma, train_mu = train_output_arti
            mdn_arti_loss = mdn.mdn_loss(train_pi, train_sigma, train_mu, b_label)
            train_mu_new = train_mu.squeeze(dim = 1)
            reconstruct_loss, pearson_loss, mse_loss = criterion_both(b_label, train_mu_new, self.out_arti_size, 50, self.device)
            loss = 0.66 * reconstruct_loss + 0.33 * mdn_arti_loss
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 2)
            self.optimizer.step()
            train_loss += loss.item()
            train_mdn_loss += mdn_arti_loss.item()
            train_pearson_loss += pearson_loss.item()
            train_mse_loss += mse_loss.item()
        return train_loss, train_mdn_loss, train_pearson_loss, train_mse_loss

    def valid(self, loader):
        valid_loss, valid_num = 0.0, 0
        for step, (batch_valid_x, batch_valid_label) in enumerate(loader):
            b_valid_x = batch_valid_x.to(self.device)
            b_valid_label = batch_valid_label.to(self.device)
            valid_num += b_valid_x.size(0)
            valid_arti_output = self.network(b_valid_x)
            valid_pi, valid_sigma, valid_mu = valid_arti_output
            valid_mdn_arti_loss = mdn.mdn_loss(valid_pi, valid_sigma, valid_mu, b_valid_label)
            valid_mu_new = valid_mu.squeeze(dim = 1)
            valid_reconstruct_loss, _, _ = criterion_both(b_valid_label, valid_mu_new, self.out_arti_size, 50, self.device)
            loss = 0.66 * valid_reconstruct_loss + 0.33 * valid_mdn_arti_loss
            valid_loss += loss.item()
        return valid_loss, valid_num

    def save_model(self):
        self.network.cpu()
        self.best_state = self.network.state_dict()
        torch.save(self.best_state, "model_parameters/best_net_inversion.pkl")
        self.network.to(self.device)

    def save_model_epoch(self, epoch):
        self.network.cpu()
        self.best_state = self.network.state_dict()
        torch.save(self.best_state, "model_parameters/inversion_net_epoch%d.pkl" % (epoch + 1))
        self.network.to(self.device)

    def train(self, epoch_num, resume_model_path = None, epoch_report = 10):
        paras_folder = "model_parameters"
        if not os.path.exists(paras_folder):
            os.system("mkdir -p %s" % paras_folder)
        if resume_model_path is not None:
            resume_model = torch.load(resume_model_path)
            self.network.load_state_dict(resume_model)
        best_valid_loss = sys.maxsize 
        for epoch in range(epoch_num):
            running_time = -time.time()
            train_loss, train_mdn_loss, train_pearson_loss, train_mse_loss = self.train_loop()
            lr_cur = self.optimizer.param_groups[0]['lr']
            running_time += time.time()
            print('Epoch: ', epoch, '| time: %2fs' % running_time, '| learning rate: %e' % lr_cur, '| train loss: %e' % train_loss, \
                '| train mdn loss: %e' % train_mdn_loss, '| train pearson loss: %e' % train_pearson_loss, '| train mse loss: %e' % train_mse_loss)
            valid_loss, valid_frames = self.valid(self.valid_loader)
            avg_valid_loss = valid_loss / valid_frames
            self.lr_scheduler.step(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                avg_best_valid_loss = best_valid_loss / valid_frames
                self.save_model()
            if epoch_report != 0:
                if (epoch + 1) % epoch_report == 0:
                    self.save_model_epoch(epoch)
            print('Epoch: ', epoch, '| best valid loss: %e' % best_valid_loss, '| valid loss: %e' % valid_loss )
            unseen_loss, unseen_frames = self.valid(self.unseen_loader)
            avg_unseen_loss = unseen_loss / unseen_frames
            seen_loss, seen_frames = self.valid(self.seen_loader)
            avg_seen_loss = seen_loss / seen_frames
            print('Epoch: ', epoch, '| unseen loss: %e' % unseen_loss, '| seen loss: %e' % seen_loss )

    def test(self, test_path, out_ark, out_scp, test_model_path = None):
        if test_model_path is None:
            test_model = torch.load("model_parameters/best_net_inversion.pkl")
        else:
            test_model = torch.load(test_model_path)
        self.network.load_state_dict(test_model)  
        self.network.eval()
        test_loader = thread_testloaddata(test_path, 1)
        test_path_list = []
        with open(test_path) as f_r:
            for line in f_r:
                test_path_list.append(line.split()[0])
        test_output_dir = "invertied_dct_output"
        if not os.path.exists(test_output_dir):
            os.system("mkdir -p %s" % test_output_dir)
        with WriteHelper("ark,scp:%s,%s" % (out_ark, out_scp)) as writer:
            for step, batch_x in enumerate(test_loader):
                utt_id = test_path_list[step]
                test_x = torch.FloatTensor(batch_x).to(self.device)
                test_output_arti = self.network(test_x)
                _, test_sigma, test_mu = test_output_arti
                mean_output = test_mu.squeeze().data.cpu().numpy()
                # sigma_output = test_sigma.squeeze().data.cpu().numpy()
                # output = np.concatenate((mean_output, sigma_output), 1)
                # assert output.shape[1] == 288
                writer(utt_id, mean_output)
