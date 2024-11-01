import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from kernels import *
from loss_helpers import *

class Clusterer:
    def __init__(self, n_clusters, n_features, lr=0.01, print_every=100, device='cuda', target_kernel = GaussianKernel(sigma=1), learned_kernel = ClusteringKernel(), loss_metric = kernel_cross_entropy, log= False):
        self.n_clusters = n_clusters
        self.lr = lr
        self.print_every = print_every
        self.device = device
        self.target_kernel = target_kernel
        self.learned_kernel = learned_kernel
        self.loss_metric = loss_metric

    def compute_loss(self, feature_map, probs):
        prob_map = self.learned_kernel(probs)
        return self.loss_metric(feature_map, prob_map)

    def fit(self, X, features, max_iters=100):
        X = X.to(self.device)
        feature_map = self.target_kernel(features)

        logits = nn.Parameter(torch.rand(X.shape[0], self.n_clusters).to(self.device))
        optimizer = optim.Adam([logits], lr=self.lr)

        loss_logs, probs_logs = [], []
        for i in range(max_iters + 1):
            optimizer.zero_grad()
            
            probs = F.softmax(logits, dim=1)
            probs_logs.append(probs.detach().cpu())

            loss = self.compute_loss(feature_map, probs)
            loss_logs.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i+1) % self.print_every == 0:
                print(f'Iteration {i + 1}/{self.max_iters}, Loss: {loss.item()}')
                self.plot(X, probs)

        return loss_logs, probs_logs, feature_map

    def plot(self, X, probs):
        clusters = probs.argmax(dim=-1).detach().cpu()
        max_probs = probs.max(dim=-1).values.detach().cpu()
        plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=clusters, cmap='tab10', alpha=max_probs)
        
        plt.show()
        plt.close()
