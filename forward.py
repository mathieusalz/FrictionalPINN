import torch
import math

def forward_original(PINN,t):       

    t = t/PINN.normalizer
    
    a = t.double() #convert to double
    for i in range(len(PINN.layers)-2):         
        z = PINN.linears[i](a)
        a = PINN.activation(z)
    a = PINN.linears[-1](a)
    
    return a

def forward_nonDim(PINN, t):         
        

    a = t.double() #convert to double
    x_proj = 2 * math.pi * a @ PINN.B.T
    x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    for i in range(len(PINN.layers)-2):         
        z = PINN.linears[i](x_proj)
        a = PINN.activation(z)
        #a = PINN.activation(PINN.ks[i] *z)
    a = PINN.linears[-1](a)
    
    return a