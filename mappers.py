import torch.nn as nn
import torch.nn.functional as F

class MLPMapper(nn.Module):
    def __init__(self, input_dim = 28*28, hidden_dims=(512, 512, 2000), output_dim = 2, probablities = False, mode = 'softmax', normalize = False):
        if hidden_dims is None:
            hidden_dims = ()
        super(MLPMapper, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend((nn.Linear(in_dim, h_dim), nn.ReLU()))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.probablities = probablities
        self.mode = mode
        self.output_dim = output_dim
        self.normalize = normalize

    def forward(self, x):
        logits = self.network(x)
        if self.probablities:
            if self.mode == 'softmax':
                logits = F.softmax(logits, dim=1)
            elif self.mode == 'cauchy':
                logits = 1 / (1 + logits**2)
                logits = F.normalize(logits, p=1, dim=1)
        if self.normalize:
            logits = F.normalize(logits, dim=1)
        return logits 
    
class IdentityMap(nn.Module):
    def forward(self, x):
        return x