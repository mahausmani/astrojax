import torch
import numpy as np
import bayesian.metrics as metrics



def validate_model(model, criterion, validloader, device, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    model.train()
    valid_loss = 0.0
    accs = []
    coverage_prob = np.array([0, 0, 0])

    for i, (x_vals, y_vals) in enumerate(validloader):

        x_vals, y_vals = x_vals.to(device), y_vals.to(device)
        outputs = torch.zeros(x_vals.shape[0], y_vals.shape[1], num_ens).to(device)
        targets = torch.zeros(y_vals.shape[0], y_vals.shape[1], num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            output, _kl = model(x_vals)
            kl += _kl
            outputs[:, :, j] = output
            targets[:, :, j] = y_vals
        
        cov_prob = metrics.coverage_prob(outputs, targets)
        coverage_prob = coverage_prob + cov_prob

        outputs = outputs.view(outputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(outputs, targets, kl, beta).item()
        accs.append(metrics.acc(outputs, targets))

    return valid_loss/len(validloader), np.mean([acc.detach().cpu().numpy() for acc in accs]), coverage_prob/len(validloader)