import torch
import torch.nn as nn
import torch.nn.functional as F 

class MCDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(MCDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        return F.dropout(x, p=self.dropout_prob, training=True, inplace=False)
    
class MCDropoutLinear(nn.Module):
    def __init__(self, inputs, outputs, hidden_dim, activation='relu', dropout=0.5):
        super(MCDropoutLinear, self).__init__()

        self.num_outputs = outputs
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        layers = []
        for idx, dim in enumerate(hidden_dim):
            if idx == 0:
                layers.append(nn.Linear(inputs, dim))
            else:
                layers.append(nn.Linear(hidden_dim[idx-1], dim))
            layers.append(self.activation)
            layers.append(MCDropout(dropout))
        layers.append(nn.Linear(hidden_dim[-1], outputs))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    