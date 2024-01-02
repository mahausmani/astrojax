import yaml
import corner
import matplotlib.pyplot as plt

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
            plot_contours=True,
            show_titles=True,
            color="orange",
            title_kwargs={"fontsize": 12}
            )
    
    if filename is not None:
        figure.savefig(filename)