import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        # Calculate the log likelihood term for Gaussian likelihood
        mse_loss = torch.mean(torch.pow(input - target, 2))
        elbo_loss = mse_loss + beta * kl

        return elbo_loss * self.train_size

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl.mean()

def acc(outputs, targets):
    return torch.mean(torch.abs(outputs-targets))

def coverage_prob(inputs, targets):
    inputs = inputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    batch_size, params, samples = inputs.shape

    param_coverage_prob = []

    for param in range(params):
        param_intervals = inputs[:, param, :]
        param_targets = targets[:, param, :]

        min_values = np.min(param_intervals, axis=1)
        max_values = np.max(param_intervals, axis=1)
        true_values = np.mean(param_targets, axis=1)

        coverage = np.sum((true_values > min_values) & (true_values < max_values)) / batch_size
        param_coverage_prob.append(coverage)
    
    return param_coverage_prob

def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta