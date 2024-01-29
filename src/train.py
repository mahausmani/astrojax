import torch
import numpy as np
import bayesian.metrics as metrics


def train_model(
    model,
    optimizer,
    criterion,
    trainloader,
    device,
    num_ens=1,
    beta_type=0.1,
    epoch=None,
    num_epochs=None,
    verbose=False,
):
    model.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    coverage_prob = np.array([0, 0, 0])
    for i, (x_vals, y_vals) in enumerate(trainloader, 1):
        optimizer.zero_grad()

        x_vals, y_vals = x_vals.to(device), y_vals.to(device)
        outputs = torch.zeros((x_vals.shape[0], y_vals.shape[1], num_ens)).to(device)
        targets = torch.zeros((y_vals.shape[0], y_vals.shape[1], num_ens)).to(device)

        kl = 0.0
        for j in range(num_ens):
            output, _kl = model(x_vals)
            kl += _kl
            outputs[:, :, j] = output
            targets[:, :, j] = y_vals
        kl = kl / num_ens
        kl_list.append(kl.item())

        cov_prob = metrics.coverage_prob(outputs, targets)
        coverage_prob = coverage_prob + cov_prob

        outputs = outputs.view(outputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        beta = metrics.get_beta(i - 1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(outputs, targets, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(outputs.data, targets))
        training_loss += loss.cpu().data.numpy()

    return (
        training_loss / len(trainloader),
        np.mean([acc.detach().cpu().numpy() for acc in accs]),
        np.mean(kl_list),
        coverage_prob / len(trainloader),
    )
