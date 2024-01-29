import numpy as np
import torch
import os
import data.dataset as dataset
import utils
import train as trn
import val
import bayesian.metrics as metrics
import argparse

from torch.optim import Adam, lr_scheduler
from bayesian.model import BayesianLinearModel
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(config_path):
    config = utils.load_config(config_path)

    datapath = config["data_path"]
    activation = config["activation"]
    train_ens = config["train_ens"]
    valid_ens = config["valid_ens"]
    samples = config["samples"]
    beta_type = config["beta_type"]
    n_epochs = config["n_epochs"]
    lr_start = config["lr_start"]
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    run_name = config["run_name"]
    train = config["train"]
    saved_model_path = config["saved_model_path"]
    labels = config["labels"]
    priors = config["priors"]

    dataloader = dataset.Dataset(datapath)

    ckpt_dir = config.get("ckpt_dir", None)
    plots_dir = config.get("plots_dir", None).format(run_name=run_name)

    ckpt_name = os.path.join(ckpt_dir, f"{run_name}.pt")

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)

    trainset, testset = dataloader.get_dataset()
    input_dim, output_dim = dataloader.get_dims()
    train_loader, valid_loader, test_loader = dataset.get_dataloader(
        trainset, testset, val_size=0.2, batch_size=batch_size
    )

    if train:
        model = BayesianLinearModel(
            inputs=input_dim,
            outputs=output_dim,
            hidden_dim=hidden_dim,
            priors=None,
            activation=activation,
        ).to(device)

        criterion = metrics.ELBO(len(trainset)).to(device)
        optimizer = Adam(model.parameters(), lr=lr_start)
        lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
        valid_loss_max = np.Inf

        for epoch in range(n_epochs):
            train_loss, train_acc, train_kl, train_coverage = trn.train_model(
                model,
                optimizer,
                criterion,
                train_loader,
                device,
                num_ens=train_ens,
                beta_type=beta_type,
                epoch=epoch,
                num_epochs=n_epochs,
            )
            valid_loss, valid_acc, val_coverage = val.validate_model(
                model,
                criterion,
                valid_loader,
                device,
                num_ens=valid_ens,
                beta_type=beta_type,
                epoch=epoch,
                num_epochs=n_epochs,
            )
            lr_sched.step(valid_loss)

            print(
                "Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \ttrain_kl_div: {:.4f}".format(
                    epoch, train_loss, train_acc, train_kl
                ),
                end="",
            )

            for i, coverage_prob in enumerate(train_coverage):
                print("\tCoverage {}: {:.4f}".format(i + 1, coverage_prob), end="")

            print(
                "\tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} ".format(
                    valid_loss, valid_acc
                ),
                end="",
            )

            for i, coverage_prob in enumerate(val_coverage):
                print("\tCoverage {}: {:.4f}".format(i + 1, coverage_prob), end="")

            # save model if validation accuracy has increased
            if valid_loss <= valid_loss_max:
                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                        valid_loss_max, valid_loss
                    )
                )

                torch.save(model.state_dict(), ckpt_name)
                valid_loss_max = valid_loss

    torch.cuda.empty_cache()
    if saved_model_path is not None:
        model = BayesianLinearModel(
            inputs=input_dim,
            outputs=output_dim,
            hidden_dim=hidden_dim,
            priors=None,
            activation=activation,
        ).to(device)
        model.load_state_dict(torch.load(saved_model_path))

    for idx, (x_vals, y_vals) in enumerate(test_loader):
        x_vals, y_vals = x_vals.to(device), y_vals.to(device)

        with torch.no_grad():
            outputs = torch.zeros(samples, x_vals.shape[0], y_vals.shape[1]).to(device)
            for j in tqdm(range(samples)):
                output, _ = model(x_vals)
                outputs[j, :, :] = output

        utils.plot(
            outputs,
            y_vals,
            labels=labels,
            filename=f"plots/bayesian/{run_name}/{idx}.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch Bayesian Neural Network")
    parser.add_argument("--config_path", type=str, help="Path to your config file")
    args = parser.parse_args()

    run(args.config_path)
