from Forward_Parameter import a, dc, b, sigma, k, eta, vpl
import torch

eps = 1.0e-30

def PDE_nonDim(PINN, velocity, state, P_t, Q_t, t):
    p = a / (velocity + eps) + eta / sigma
    r = 1.0 - velocity * state / dc
    q = k / sigma * (vpl - velocity) - b * r / (state + eps)

    f_P = P_t - q / (p * velocity) * PINN.Tc
    f_Q = Q_t - r / state * PINN.Tc

    Lr = (f_P ** 2 + f_Q ** 2)[:, 0]

    loss_f = PINN.reduction(Lr)
    loss_f = loss_f / 2 

    return loss_f

def PDE_nonDim_mixed(PINN, velocity, state, P_t, Q_t, t):
    p = a / (velocity + eps) + eta / sigma
    r = 1.0 - velocity * state / dc
    q = k / sigma * (vpl - velocity) - b * r / (state + eps)

    f_P = (1/PINN.Tc * P_t - q / (p * velocity)) * PINN.nonDimensionalizer
    f_Q = (1/PINN.Tc * Q_t - r / state) * PINN.nonDimensionalizer

    Lr = (f_P ** 2 + f_Q ** 2)[:, 0]

    loss_f = PINN.reduction(Lr)
    loss_f = loss_f / 2 

    return loss_f

def PDE_original(PINN, velocity, state, P_t, Q_t, t):
    p = a / (velocity + eps) + eta / sigma
    r = 1.0 - velocity * state / dc
    q = k / sigma * (vpl - velocity) - b * r / (state + eps)

    f_P = P_t - q / (p * velocity)
    f_Q = Q_t - r / state

    Lr = (f_P ** 2 + f_Q ** 2)[:, 0]
    t_ = t[:,0]
    loss_f = Lr[0]*(t_[1]-t_[0]) + Lr[-1]*(t_[-1]-t_[-2]) + torch.sum(Lr[1:-1]*(t_[2:]-t_[0:-2]))
    loss_f = loss_f / 2 * (dc / vpl)
    
    return loss_f 