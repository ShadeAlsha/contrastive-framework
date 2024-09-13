import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class Kernel(nn.Module):
    def __init__(self):
        super(Kernel, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Kernel):
            return CompositeKernel([self, other], operation='add')
        elif isinstance(other, (int, float)):
            return ConstantAddedKernel(self, other)
        else:
            raise ValueError("Unsupported addition with non-kernel or non-scalar type.")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Kernel):
            return CompositeKernel([self, other], operation='sub')
        elif isinstance(other, (int, float)):
            return ConstantAddedKernel(self, -other)
        else:
            raise ValueError("Unsupported subtraction with non-kernel or non-scalar type.")

    def __mul__(self, scalar):
        return ScaledKernel(self, scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __max__(self, other):
        return CompositeKernel([self, other], operation='max')

    def leak_kernel(self, alpha):
        return LeakKernel(self, alpha)

    def normalize(self):
        return NormalizedKernel(self)

class CompositeKernel(Kernel):
    def __init__(self, kernels, operation='add'):
        super(CompositeKernel, self).__init__()
        self.kernels = kernels
        self.operation = operation

    def forward(self, *args, **kwargs):
        if self.operation == 'add':
            return sum(kernel(*args, **kwargs) for kernel in self.kernels)
        elif self.operation == 'sub':
            result = self.kernels[0](*args, **kwargs)
            for kernel in self.kernels[1:]:
                result -= kernel(*args, **kwargs)
            return result
        elif self.operation == 'compose':
            result = self.kernels[0](*args, **kwargs)
            for kernel in self.kernels[1:]:
                result = kernel(result)
            return result
        elif self.operation == 'max':
            return torch.max(torch.stack([kernel(*args, **kwargs) for kernel in self.kernels]), dim=0)[0]

class ScaledKernel(Kernel):
    def __init__(self, kernel, scalar):
        super(ScaledKernel, self).__init__()
        self.kernel = kernel
        self.scalar = scalar

    def forward(self, *args, **kwargs):
        return self.scalar * self.kernel(*args, **kwargs)

class LeakKernel(Kernel):
    def __init__(self, kernel, alpha):
        super(LeakKernel, self).__init__()
        self.kernel = kernel
        self.alpha = alpha

    def forward(self, *args, **kwargs):
        kernel_output = self.kernel(*args, **kwargs)
        num_neighbors = kernel_output.size(1)
        leaked_output = (1 - self.alpha) * kernel_output + self.alpha / num_neighbors
        return leaked_output

class ConstantAddedKernel(Kernel):
    def __init__(self, kernel, constant):
        super(ConstantAddedKernel, self).__init__()
        self.kernel = kernel
        self.constant = constant

    def forward(self, *args, **kwargs):
        return self.kernel(*args, **kwargs) + self.constant

class NormalizedKernel(Kernel):
    def __init__(self, kernel):
        super(NormalizedKernel, self).__init__()
        self.kernel = kernel

    def forward(self, *args, **kwargs):
        kernel_output = self.kernel(*args, **kwargs)
        row_sums = kernel_output.sum(dim=1, keepdim=True)
        normalized_output = kernel_output / row_sums
        return normalized_output


class ClusteringKernel(Kernel):
    def forward(self, cluster_probabilities, labels = None, idx = None):
        cluster_sizes = cluster_probabilities.sum(dim=0)
        neighbors_prob = (cluster_probabilities / cluster_sizes) @ cluster_probabilities.t()
        return neighbors_prob
        
class DistancesKernel(Kernel):
    def __init__(self, type='euclidean'):
        super(DistancesKernel, self).__init__()
        self.type = type

    def forward(self, features, labels = None, idx = None):
        if self.type == 'euclidean':
            distances = torch.cdist(features, features, p=2)
        elif self.type == 'cosine':
            features = F.normalize(features, p=2, dim=1)
            distances = -features @ features.T
        elif self.type == 'dot':
            distances = -features @ features.T
        return distances

class GaussianKernel(Kernel):
    def __init__(self, sigma, type='euclidean', mask_diagonal=True):
        super(GaussianKernel, self).__init__()
        self.sigma = sigma
        self.type = type
        self.mask_diagonal = mask_diagonal

    def forward(self, features, labels = None, idx = None):
        distances = DistancesKernel(type=self.type)(features)
        if self.mask_diagonal:
            mask = 1 - torch.eye(features.size(0), device=features.device)
            distances.masked_fill(mask == 0, -float('inf'))
        neighbors_prob = F.softmax(-distances / self.sigma**2, dim=1)
        return neighbors_prob
        
class GaussianKernelwLearnedSigmas(Kernel):
    def __init__(self, type='euclidean', mask_diagonal=True):
        super(GaussianKernelwLearnedSigmas, self).__init__()
        self.type = type
        self.mask_diagonal = mask_diagonal
        self.sigmas = nn.Embeddings()

    def forward(self, features, probs):
        distances = DistancesKernel(type=self.type)(features)

        if self.mask_diagonal:
            mask = 1 - torch.eye(features.size(0), device=features.device)
            distances.masked_fill(mask == 0, -float('inf'))

        per_sample_sigma = probs@self.sigmas
        sigmas_expanded = per_sample_sigma.unsqueeze(1)
        
        neighbors_prob = F.softmax(-distances / (sigmas_expanded ** 2), dim=1)
        return neighbors_prob

class PartialGaussianKernel(GaussianKernel):
    def __init__(self, sigma, mask, type='euclidean'):
        super(PartialGaussianKernel, self).__init__(sigma, type=type)
        self.mask = mask

    def forward(self, features, labels = None, idx = None):
        distances = DistancesKernel(type=self.type)(features)
        distances.masked_fill(self.mask == 0, -float('inf'))
        neighbors_prob = F.softmax(-distances / self.sigma**2, dim=1)
        return neighbors_prob

class CauchyKernel(Kernel):
    def __init__(self, gamma, type='euclidean', mask_diagonal=True):
        super(CauchyKernel, self).__init__()
        self.gamma = gamma
        self.type = type
        self.mask_diagonal = mask_diagonal

    def forward(self,  features, labels = None, idx = None):
        distances = DistancesKernel(type=self.type)(features)
        neighbors_prob_unormalized = 1 / (1 + (distances / self.gamma)**2)
        if self.mask_diagonal:
            mask = 1 - torch.eye(neighbors_prob_unormalized.size(0), device=neighbors_prob_unormalized.device)
            neighbors_prob_unormalized.masked_fill(mask == 0, 0)
        neighbors_prob = F.normalize(neighbors_prob_unormalized, p=1, dim=1)
        return neighbors_prob

class KnnKernel(Kernel):
    def __init__(self, k, type='euclidean'):
        super(KnnKernel, self).__init__()
        self.k = k
        self.type = type

    def forward(self,  features, labels = None, idx = None):
        distances = DistancesKernel(type=self.type)(features)
        return knn_from_dist(distances, self.k)

class LabelsKernel(Kernel):
    def forward(self, features = None, labels = None, idx = None):
        N = labels.shape[0]
        adj_matrix = torch.zeros((N, N), device=labels.device)
        ll = labels.unsqueeze(0)
        adj_matrix[ll == ll.T] = 1
        neighbors_prob = F.normalize(adj_matrix, p=1, dim=1)
        return neighbors_prob

class AugmentationKernel(Kernel):
    def forward(self, features = None, labels = None, idx = None):
        N = idx.shape[0]
        adj_matrix = torch.zeros((N, N), device=labels.device)
        ll = idx.unsqueeze(0)
        adj_matrix[ll == ll.T] = 1
        neighbors_prob = F.normalize(adj_matrix, p=1, dim=1)
        return neighbors_prob
    
def knn_from_dist(dist_map, k):
    N = dist_map.shape[0]
    adj_matrix = torch.zeros_like(dist_map)

    knn_indices = torch.topk(dist_map, k + 1, largest=False, dim=1).indices[:, 1:]
    batch_indices = torch.arange(N).repeat_interleave(k)
    adj_matrix[batch_indices, knn_indices.reshape(-1)] = 1

    neighbors_prob = F.normalize(adj_matrix, p=1, dim=1)
    return neighbors_prob
