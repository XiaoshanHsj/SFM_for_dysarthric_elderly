from __future__ import division
import numpy as np
import torch

def criterion_pearson(y, y_pred, out_arti_size, device):
    epsilon = 1e-6
    y = y.reshape((1, -1, out_arti_size))
    y_pred = y_pred.reshape((1, -1, out_arti_size))
    y_1 = y.sub(torch.mean(y, dim=1, keepdim=True))
    y_pred_1 = y_pred.sub(torch.mean(y_pred,dim=1, keepdim=True))
    nume = torch.sum(y_1 * y_pred_1, dim=1, keepdim=True)
    deno = torch.sqrt(torch.sum(y_1 ** 2, dim=1, keepdim=True)) * \
        torch.sqrt(torch.sum(y_pred_1 ** 2, dim=1, keepdim=True))
    minim = torch.tensor(epsilon,dtype=torch.float64)  # avoid division by 0
    minim = minim.to(device=device)
    deno = deno.to(device=device)
    nume = nume.to(device=device)
    nume = nume + minim
    deno = deno + minim
    my_loss = torch.div(nume, deno)  # (B,1,18)
    my_loss = torch.sum(my_loss)
    return -my_loss

def criterion_both(my_y,my_ypred,out_arti_size,alpha,device):
    compl = torch.tensor(1. - float(alpha) / 100., dtype=torch.float64)
    alpha = torch.tensor(float(alpha) / 100., dtype = torch.float64)
    multip = torch.tensor(float(1), dtype = torch.float64)
    alpha = alpha.to(device = device)
    multip = multip.to(device = device)
    compl = compl.to(device= device)
    a = alpha * criterion_pearson(my_y, my_ypred, out_arti_size, device) * multip
    b = compl * torch.nn.MSELoss(reduction='mean')(my_y, my_ypred).to(device) * 10
    new_loss = a + b
    return new_loss, a, b
