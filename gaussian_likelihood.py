import torch
import numpy as np

def gaussian_likelihood(action, mu, log_std):
    EPS = 1e-8
    
    log_likelihood = -0.5 * (((action - mu) / (torch.exp(log_std) + EPS))**2 + 2 * log_std + np.log(2 * torch.pi))
    return log_likelihood.sum(axis=-1)
    