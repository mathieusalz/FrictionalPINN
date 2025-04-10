import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.autograd as autograd
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy import format_float_scientific as ffs
import sys
import time as Time
from Forward_Parameter import a, dc, b, sigma, k, eta, vpl, statepl, v_ini, state_ini
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=5)
from PDE_funcs import PDE_nonDim, PDE_original, PDE_nonDim_mixed
from forward import forward_original, forward_nonDim
from typing_extensions import Literal


config_types = Literal['original', 
                       'non_dim_phys',
                       'non_dim_norm',
                       'original_nondim1',
                       'original_nondim2',
                       'non_dim_mixed1',
                       'non_dim_mixed2']

class Sequentialmodel(nn.Module):
    
    def __init__(self, 
                 layers, 
                 collocation_points, 
                 config : config_types = "original", 
                 reduction = torch.mean,
                 activation = nn.Tanh(), 
                 device = 'cpu'):
        super().__init__()
        self.activation = activation
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        # Define a learnable k per layer (except last layer, no activation)
        # self.ks = nn.ParameterList([
        #     nn.Parameter(torch.tensor(1.0, dtype=torch.double), requires_grad=True)
        #     for _ in range(len(layers) - 2)
        # ])

        # self.ks = nn.ParameterList([nn.Parameter(torch.tensor(1.0, dtype=torch.double), requires_grad=True)])
        # self.ks = nn.ParameterList()

        # for i in range(len(layers) - 1):

        #     # Only hidden layers have k parameters (exclude last layer)
        #     if i < len(layers) - 2:
        #         k = nn.Parameter(torch.ones(layers[i + 1], dtype=torch.double), requires_grad=True)
        #         self.ks.append(k)

        self.B = nn.Parameter(
            torch.randn(3, 1) * 1,
            requires_grad=False  # fixed random projection
        )

        self.iter = 0
        
        self.loss_hist = []
        self.lossini_hist = []
        self.lossf_hist = []
        self.device = device
        self.layers = layers
        self.config = config
        self.reduction = reduction
        self.verbose = True

        self.time = torch.from_numpy((collocation_points).reshape((-1, 1))).double().to(device)

        self.time_span = self.time[-1][0] - self.time[0][0]

        # Original Formulation from Paper
        # Normalize time: using forward func
        # Nondim PDE: after MSE
        if self.config == 'original':
            self.normalizer = self.time_span
            self.nonNormalizer = 1
            self.pde_loss = PDE_original
            self.forward_func = forward_original

        # Normalize time: divide by Tc
        # Nondim PDE: multiply by Tc
        elif self.config == 'non_dim_phys':
            self.normalizer = 1
            self.Tc = dc/vpl
            self.nonNormalizer = self.Tc
            self.time = self.time/self.Tc
            self.pde_loss = PDE_nonDim
            self.forward_func = forward_nonDim
        
        # Normalize time: divide by Tc
        # Nondim PDE: multiply by Tc
        elif self.config == 'non_dim_norm':
            self.normalizer = 1
            self.Tc = self.time_span
            self.nonNormalizer = self.Tc
            self.time = self.time/self.Tc
            self.pde_loss = PDE_nonDim
            self.forward_func = forward_nonDim

        # Normalize time: using forward func
        # Nondim PDE: multiply by Tc
        elif self.config == 'original_nondim1':
            self.normalizer = self.time_span
            self.nonNormalizer = 1
            self.Tc = dc/vpl
            self.pde_loss = PDE_nonDim
            self.forward_func = forward_original

        # Normalize time: using forward func
        # Nondim PDE: multiply by time span
        elif self.config == 'original_nondim2':
            self.normalizer = self.time_span
            self.nonNormalizer = 1
            self.Tc = self.time_span
            self.pde_loss = PDE_nonDim
            self.forward_func = forward_original

        # Normalize time: divide by time span
        # Nondim PDE: multiply by Tc 
        elif self.config == 'non_dim_mixed1':
            self.normalizer = 1
            self.Tc = dc/vpl
            self.nonDimensionalizer = self.time_span
            self.nonNormalizer = self.Tc
            self.time = self.time/self.Tc
            self.pde_loss = PDE_nonDim_mixed
            self.forward_func = forward_nonDim

        # Normalize time: divide by Tc
        # Nondim PDE:multiply by time span
        elif self.config == 'non_dim_mixed2':
            self.normalizer = 1
            self.Tc = self.time_span
            self.nonDimensionalizer = dc/vpl
            self.nonNormalizer = self.time_span
            self.time = self.time/self.time_span
            self.pde_loss = PDE_nonDim_mixed
            self.forward_func = forward_nonDim
        
        self.Nt = self.time.shape[0]
        self.u_ini = torch.from_numpy(np.hstack((np.log(np.array([v_ini]) / vpl), np.log(np.array([state_ini]) / statepl))))

        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def set_verbose(self, state):
        self.verbose = state

    def forward(self, z):
        return self.forward_func(self, z)

    def forward_by_layer(self, t, layer_num):

        a = t.double() #convert to double
        for i in range(layer_num):         
            z = self.linears[i](a)
            a = self.activation(z)
        
        return a



    #loss function induced from initial condition
    def loss_IC(self):

        z = self.time[0,:].clone()
        z.requires_grad = True
        
        u = self.forward(z)

        loss_u = self.loss_function(u, self.u_ini)#velocity and state

        return loss_u

    #loss function induced from governing equation
    def loss_PDE(self):
                       
        z = self.time.clone()
        z.requires_grad = True
        
        u = self.forward(z)
        
        u_0 = torch.reshape(u[:,0], (self.Nt,1)) #v
        u_1 = torch.reshape(u[:,1], (self.Nt,1)) #state
        
        P_t = autograd.grad(u_0, z,torch.ones([self.Nt, 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        Q_t = autograd.grad(u_1, z,torch.ones([self.Nt, 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        
        velocity = vpl * torch.exp(u_0)
        state    = statepl * torch.exp(u_1)
                
        loss_f = self.pde_loss(self, velocity, state, P_t, Q_t, self.time)
                
        return loss_f

    def loss(self):
        
        loss_ini = self.loss_IC()
        loss_f = self.loss_PDE()
        loss_val = loss_ini + loss_f

        return loss_val, loss_ini, loss_f
                                      
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def closure(self):
        
        self.optimizer.zero_grad()
        loss, loss_ini, loss_f = self.loss()
        self.loss_hist.append(loss.item())
        self.lossini_hist.append(loss_ini.item())
        self.lossf_hist.append(loss_f.item())
        loss.backward()        
        self.iter += 1
        
        if self.verbose:
            if self.iter <= 9:
                _ = self.test()
                print(f'Step: {self.iter} \t Total loss: {loss.item():.4e}  \t Initial Loss: {loss_ini.item():.4e} \t  PDE Loss: {loss_f.item():4e}')

            if self.iter % 100 == 0:
                _ = self.test()
                print(f'Step: {self.iter} \t Total loss: {loss.item():.4e}  \t Initial Loss: {loss_ini.item():.4e} \t  PDE Loss: {loss_f.item():4e}')

        return loss
    
    def test(self):
                
        u_pred = self.forward(self.time)
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred,(self.Nt,2),order='F')
                
        return u_pred