import torch

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
    for i in range(len(PINN.layers)-2):         
        z = PINN.linears[i](a)
        a = PINN.activation(z)
    a = PINN.linears[-1](a)
    
    return a