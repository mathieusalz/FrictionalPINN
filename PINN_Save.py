import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.autograd as autograd
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
from numpy import format_float_scientific as ffs

device = 'cpu'

def Save_Loss_forward(PINN, loss_list, i, deep = False):

    loss_list["total"].append(PINN.loss_hist[-1])
    loss_list["ini"].append(PINN.lossini_hist[-1]) 
    loss_list["ode"].append(PINN.lossf_hist[-1])
    
    if i%100==0:
        u_pred = PINN.test()
        loss_list["p"].append(u_pred[:,0])
        loss_list["q"].append(u_pred[:,1])
        loss_list["iter"].append(i)

        if deep:
            for key in loss_list['layer_activations'].keys():
                loss_list['layer_activations'][key].append(PINN.forward_by_layer(PINN.time,int(key)))

def Save_PINN_forward(PINN, name, loss_list):
    model = PINN.to(device)
    torch.save(model.state_dict(), name + '.pth')

    f = h5py.File(name + '_loss.h5', 'w')
    f["loss"] = loss_list["total"]
    f["loss_ini"] = loss_list["ini"]
    f["loss_f"] = loss_list["ode"]
    f['p'] = loss_list['p']
    f['q'] = loss_list['q']
    f.close()

    print("NN is saved")