import os
import yaml
import corner
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

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