import torch.nn as nn
from .layers import BayesianLinear
from .misc import ModuleWrapper

class BayesianLinearModel(ModuleWrapper):
    def __init__(self, inputs, outputs, hidden_dim, priors=None, activation='relu'):
        
        super(BayesianLinearModel, self).__init__()

        self.num_outputs = outputs
        self.priors = priors
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        layers = []
        for idx, dim in enumerate(hidden_dim):
            if idx == 0:
                layers.append(BayesianLinear(inputs, dim, priors=priors))
            else:
                layers.append(BayesianLinear(hidden_dim[idx-1], dim, priors=priors))
            layers.append(self.activation)
        layers.append(BayesianLinear(hidden_dim[-1], outputs, priors=priors))

        self.layers = nn.Sequential(*layers)

    # def forward(self, x):
    #     return self.layers(x)


        
