import os
import corner
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def plot(outputs, targets, labels, filename=None):
    """Plot corner plot of the outputs and targets"""
    outputs = outputs.cpu().detach().numpy()
    outputs = outputs.reshape(-1, outputs.shape[-1])
    targets = targets.cpu().detach().numpy()
    targets = targets.flatten()

    # print(outputs.shape, targets.shape)
    
    figure = corner.corner(
            outputs,
            labels=labels, 
            truths=targets,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
            )
    
    if filename is not None:
        figure.savefig(filename)