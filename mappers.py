import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPMapper(nn.Module):
    def __init__(self, input_dim = 28*28,
                       hidden_dims = [512, 256, 128],
                       output_dim = 2,
                       softmax = False):
        super(MLPMapper, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.softmax = softmax

    def forward(self, x):
        logits = self.network(x)
        if self.softmax:
            logits = F.softmax(logits, dim=1)
        return logits 