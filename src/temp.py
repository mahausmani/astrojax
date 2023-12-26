import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from torch.optim import Adam, lr_scheduler
from model import BayesianLinearModel
import dataset
import utils
import metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None, verbose=False):
    model.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (x_vals, y_vals) in enumerate(trainloader, 1):
        optimizer.zero_grad()

        x_vals, y_vals = x_vals.to(device), y_vals.to(device)
        outputs = torch.zeros((x_vals.shape[0], y_vals.shape[1], num_ens)).to(device)

        kl = 0.0
        for j in range(num_ens):
            output, _kl = model(x_vals)
            kl += _kl
            outputs[:, :, j] = output   

        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)
        
        beta = metrics.get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, y_vals, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, y_vals))
        training_loss += loss.cpu().data.numpy()

    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)

def validate_model(model, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    model.train()
    valid_loss = 0.0
    accs = []

    for i, (x_vals, y_vals) in enumerate(validloader):
        x_vals, y_vals = x_vals.to(device), y_vals.to(device)
        outputs = torch.zeros(x_vals.shape[0], y_vals.shape[1], num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            output, _kl = model(x_vals)
            kl += _kl
            outputs[:, :, j] = output

        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, y_vals, kl, beta).item()
        accs.append(metrics.acc(log_outputs, y_vals))

    return valid_loss/len(validloader), np.mean([acc.detach().numpy() for acc in accs])

def run(datapath, run_num, train):
    dataloader = dataset.Dataset(data_path=datapath)

    activation = 'relu'
    train_ens = 1
    valid_ens = 1
    samples = 1000
    beta_type = 0.1
    n_epochs = 100
    lr_start = 0.001
    batch_size = 15
    hidden_dim = [1000, 1000, 1000, 1000]

    # priors = {
    # 'prior_mu': 0,
    # 'prior_sigma': 0.1,
    # 'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    # 'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
    # }

    ckpt_dir = f'checkpoints/bayesian'
    ckpt_name = f'checkpoints/bayesian/model_{run_num}.pt'

    plots_dir = f'plots/bayesian/{run_num}'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)

    trainset, testset = dataloader.get_dataset()
    input_dim, output_dim = dataloader.get_dims()
    train_loader, valid_loader, test_loader = dataset.get_dataloader(trainset, testset, val_size=0.2, batch_size=batch_size)
    if train:
        model = BayesianLinearModel(inputs=input_dim, outputs=output_dim, hidden_dim=hidden_dim, priors=None, activation=activation).to(device)

        criterion = metrics.ELBO(len(trainset)).to(device)
        optimizer = Adam(model.parameters(), lr=lr_start)
        lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
        valid_loss_max = np.Inf

        for epoch in range(n_epochs):

            train_loss, train_acc, train_kl = train_model(model, optimizer, criterion, train_loader, num_ens=train_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
            valid_loss, valid_acc = validate_model(model, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
            lr_sched.step(valid_loss)

            print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

            # save model if validation accuracy has increased
            if valid_loss <= valid_loss_max:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_max, valid_loss))
                
                torch.save(model.state_dict(), ckpt_name)
                valid_loss_max = valid_loss

    labels = [r'$\alpha$', r'$mass_{min}$', r'$mass_{max}$', r'$Mass_{max}$', r'$\sigma_{ecc}$']
    for idx, (x_vals, y_vals) in enumerate(test_loader):
        x_vals, y_vals = x_vals.to(device), y_vals.to(device)
        outputs = torch.zeros(samples, x_vals.shape[0], y_vals.shape[1]).to(device)
        for j in range(samples):
            output, _ = model(x_vals)
            outputs[j, :, :] = output

        fig = utils.plot(outputs, y_vals, labels=labels, filename = f'plots/bayesian/{run_num}/{idx}.png')
        

run('/home/safi/Semester 07/Kaavish/bnn/data', run_num="0.0.1", train=True)
