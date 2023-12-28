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
        log_likelihood = -0.5 * (torch.log(2 * torch.tensor(math.pi)) + torch.pow(target - input, 2))
        log_likelihood = torch.sum(log_likelihood, dim=1)
        elbo_loss = torch.mean(log_likelihood) + beta * kl

        return elbo_loss / self.train_size

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def acc(outputs, targets):
    """Computes the accuracy for multiple binary predictions"""
    return torch.mean(torch.abs(outputs-targets))

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