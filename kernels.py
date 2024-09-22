import torch
import torch.nn as nn
import torch.nn.functional as F

################## Kernels Base Class ##################
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

    def leak(self, alpha):
        return LeakKernel(self, alpha)

    def normalize(self):
        return NormalizedKernel(self)
    
    def binarize(self):
        return BinarizedKernel(self)
    
    def neighbor_propagation(self, steps=1):
        return NeighborPropagationKernel(self, steps=steps)

################## Kernel Operations ##################
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

class BinarizedKernel(Kernel):
    def __init__(self, base_kernel):
        super(BinarizedKernel, self).__init__()
        self.base_kernel = base_kernel

    def forward(self, *args, **kwargs):
        # Compute the base kernel output
        kernel_output = self.base_kernel(*args, **kwargs)
        
        # Map all non-zero elements to 1 and zero elements to 0
        binarized_output = (kernel_output > 0).float()
        
        return binarized_output

class MaxKernel(Kernel):
    def __init__(self, kernel1, kernel2):
        super(MaxKernel, self).__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def forward(self, *args, **kwargs):
        output1 = self.kernel1(*args, **kwargs)
        output2 = self.kernel2(*args, **kwargs)
        
        return torch.max(output1, output2)

class NeighborPropagationKernel(Kernel):
    def __init__(self, base_kernel, steps=1):
        super(NeighborPropagationKernel, self).__init__()
        self.base_kernel = base_kernel
        self.steps = steps

    def forward(self, *args, **kwargs):
        # Compute the base adjacency matrix
        adj_matrix = self.base_kernel(*args, **kwargs)
        adj_matrix = (adj_matrix+ adj_matrix.T)/2
        
        # Apply neighbor propagation for the specified number of steps
        smooth_adj_matrix = adj_matrix.clone()
        for _ in range(self.steps):
            smooth_adj_matrix = torch.mm(smooth_adj_matrix, adj_matrix)
            smooth_adj_matrix = (smooth_adj_matrix > 0).float()

        return smooth_adj_matrix



################## Distance Kernels ##################
class DistancesKernel(Kernel):
    def __init__(self, type='euclidean'):
        super(DistancesKernel, self).__init__()
        self.type = type

    def forward(self, features, labels=None, idx=None):
        if isinstance(features, list) and len(features) == 2:
            online_features, ema_features = features[0], features[1]
        else:
            online_features, ema_features = features, features

        if self.type == 'euclidean':
            distances = torch.cdist(online_features, ema_features, p=2)
        elif self.type == 'cosine':
            online_features = F.normalize(online_features, p=2, dim=1)
            ema_features = F.normalize(ema_features, p=2, dim=1)
            distances = -online_features @ ema_features.T
        elif self.type == 'dot':
            distances = -online_features @ ema_features.T
        return distances

class GaussianKernel(Kernel):
    def __init__(self, sigma=None, sigma_grid=None, perplexity=None, mode='sigma', type='euclidean', mask_diagonal=True, symmetric=True):
        super(GaussianKernel, self).__init__()
        assert mode in ['sigma', 'perplexity_binary_search', 'perplexity_heuristic', 'perplexity_grid_search'], \
            "Mode must be either 'sigma', 'perplexity_binary_search', 'perplexity_heuristic', or 'perplexity_grid_search'"
        
        if mode == 'sigma' and sigma is None:
            raise ValueError("Sigma must be provided when mode is 'sigma'")
        if mode in ['perplexity_binary_search', 'perplexity_heuristic', 'perplexity_grid_search'] and perplexity is None:
            raise ValueError("Perplexity must be provided when mode is 'perplexity_binary_search', 'perplexity_heuristic', or 'perplexity_grid_search'")
        
        self.sigma = sigma
        self.sigma_grid = sigma_grid
        self.perplexity = perplexity
        self.mode = mode
        self.type = type
        self.mask_diagonal = mask_diagonal
        self.symmetric = symmetric

    def forward(self, features, labels=None, idx=None, return_log=False):
        # Step 1: Compute pairwise distances
        distances = DistancesKernel(type=self.type)(features)

        if self.mask_diagonal:
            mask = 1 - torch.eye(distances.size(0), device=distances.device)
            distances.masked_fill(mask == 0, float('inf'))

        # Step 2: Initialize sigma grid if needed
        if self.mode == 'perplexity_grid_search' and self.sigma_grid is None:
            d_median = torch.median(distances[distances != float('inf')])
            self.sigma_grid = torch.logspace(-2, 2, steps=100, device=distances.device) * d_median

        # Step 3: Select the mode and compute probabilities
        if self.mode == 'perplexity_binary_search':
            self.sigma = self.find_sigma_with_perplexity_binary_search(distances)
        elif self.mode == 'perplexity_heuristic':
            self.sigma = self.find_sigma_with_perplexity_heuristic(distances, k=int(self.perplexity))
        elif self.mode == 'perplexity_grid_search':
            self.sigma = self.find_best_sigma_grid_search_for_each_point(distances)

        neighbors_prob = self.compute_probabilities_with_sigma(distances, return_log)
        
        if self.symmetric and not return_log:
            neighbors_prob = (neighbors_prob + neighbors_prob.T) / 2
        return neighbors_prob

    def find_best_sigma_grid_search_for_each_point(self, distances):
        """Grid search over sigma values to match the desired perplexity for each point."""
        probabilities = self.compute_probabilities_for_sigma_grid(distances)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=2)  # Compute entropy for each point and each sigma
        perplexity_diff = torch.abs(torch.exp(entropy) - self.perplexity)
        best_sigma_idx = torch.argmin(perplexity_diff, dim=1)  # Find best sigma for each point
        return self.sigma_grid[best_sigma_idx]

    def compute_probabilities_for_sigma_grid(self, distances):
        """Compute probabilities for all sigma values in the grid for each point."""
        distances = distances.unsqueeze(-1)  # Add an extra dimension for broadcasting
        sigmas_squared = self.sigma_grid**2
        exp_distances = torch.exp(-distances / sigmas_squared)
        probabilities = exp_distances / torch.sum(exp_distances, dim=1, keepdim=True)
        return probabilities  # Shape: (N, N, num_sigma)

    def compute_probabilities_with_sigma(self, distances, return_log=False):
        """Compute Gaussian similarities using the selected sigma for each point."""
        sigmas = self.sigma#.unsqueeze(1)  # Shape: (N, 1)
        if return_log:
            neighbors_prob = torch.log_softmax(-distances / sigmas**2, dim=1)
        else:
            neighbors_prob = F.softmax(-distances / sigmas**2, dim=1)
        return neighbors_prob


    def find_sigma_with_perplexity_binary_search(self, distances):
        n = distances.size(0)
        P = torch.zeros_like(distances)
        beta = torch.ones(n, device=distances.device)  # Beta is 1 / (2 * sigma^2)
        log_perplexity = torch.log(torch.tensor(self.perplexity, device=distances.device))

        # Binary search for each point to find the appropriate sigma (via beta)
        for i in range(n):
            beta_i = beta[i]
            this_distances = distances[i, :]
            this_distances = this_distances[this_distances != float('inf')]  # Remove infs from diagonal

            # Binary search for the right beta (or sigma)
            betamin, betamax = -float('inf'), float('inf')
            for _ in range(50):  # Max 50 iterations for binary search
                P_i = F.softmax(-this_distances * beta_i, dim=0)
                entropy = -torch.sum(P_i * torch.log(P_i + 1e-10))
                current_perplexity = torch.exp(entropy)

                if torch.abs(current_perplexity - self.perplexity) < 1e-5:
                    break  # Converged
                
                if current_perplexity > self.perplexity:
                    betamin = beta_i
                    beta_i = (beta_i + betamax) / 2 if betamax != float('inf') else beta_i * 2
                else:
                    betamax = beta_i
                    beta_i = (beta_i + betamin) / 2 if betamin != -float('inf') else beta_i / 2

            beta[i] = beta_i
            P[i, :this_distances.size(0)] = F.softmax(-this_distances * beta_i, dim=0)

        return P

    def find_sigma_with_perplexity_heuristic(self, distances, k=None):
        n = distances.size(0)
        
        # Set k as the perplexity if not provided
        if k is None:
            k = int(self.perplexity)

        # Sort distances and get the top-k nearest neighbors for each point
        sorted_distances, _ = torch.sort(distances, dim=1)

        # Select the k-nearest neighbors (excluding the first one which is the point itself)
        k_nearest_neighbors = sorted_distances[:, 1:k+1]  # Exclude diagonal (self-distance)
        
        # Compute sigma as the mean of the k-nearest neighbors distances for each point
        sigma = torch.mean(k_nearest_neighbors, dim=1, keepdim=True)

        return sigma

class CauchyKernel(Kernel):
    def __init__(self, gamma, type='euclidean', mask_diagonal=True, symmetric=False, normalize=True):
        super(CauchyKernel, self).__init__()
        self.gamma = gamma
        self.type = type
        self.mask_diagonal = mask_diagonal
        self.symmetric = symmetric
        self.normalize = normalize

    def forward(self,  features, labels = None, idx = None, return_log = False):
        distances = DistancesKernel(type=self.type)(features)
        neighbors_prob_unormalized = 1 / (1 + (distances / self.gamma)**2)
        if self.mask_diagonal:
            mask = 1 - torch.eye(neighbors_prob_unormalized.size(0), device=neighbors_prob_unormalized.device)
            neighbors_prob_unormalized.masked_fill(mask == 0, 0)
        if self.normalize:
            neighbors_prob = (
                torch.log(neighbors_prob_unormalized)
                - torch.log(neighbors_prob_unormalized.sum(dim=1, keepdim=True))
                if return_log
                else F.normalize(neighbors_prob_unormalized, p=1, dim=1)
            )
        elif return_log:
            neighbors_prob = torch.log(neighbors_prob_unormalized)
        else:
            neighbors_prob = neighbors_prob_unormalized
        if self.symmetric and not return_log and not self.normalize:
            neighbors_prob = (neighbors_prob + neighbors_prob.T) / 2
        return neighbors_prob

################## Graph Kernels ##################

class KnnKernel(Kernel):
    def __init__(self, k, type='euclidean'):
        super(KnnKernel, self).__init__()
        self.k = k
        self.type = type

    def forward(self, features, labels=None, idx=None):
        distances = DistancesKernel(type=self.type)(features)
        adj_matrix = knn_from_dist(distances, self.k)
        neighbors_prob = F.normalize(adj_matrix, p=1, dim=1)
        return neighbors_prob

class LabelsKernel(Kernel):
    def forward(self, features=None, labels=None, idx=None):
        N = labels.shape[0]
        adj_matrix = torch.zeros((N, N), device=labels.device)
        ll = labels.unsqueeze(0)
        adj_matrix[ll == ll.T] = 1
        
        # Set diagonal to zero to remove self-connections
        adj_matrix.fill_diagonal_(0)
        
        # Normalize the adjacency matrix
        neighbors_prob = F.normalize(adj_matrix, p=1, dim=1)
        return neighbors_prob

class AugmentationKernel(Kernel):
    def forward(self, features=None, labels=None, idx=None):
        N = idx.shape[0]
        adj_matrix = torch.zeros((N, N), device=idx.device)
        ll = idx.unsqueeze(0)
        adj_matrix[ll == ll.T] = 1
        
        # Set diagonal to zero to remove self-connections
        adj_matrix.fill_diagonal_(0)
        
        # Normalize the adjacency matrix
        neighbors_prob = F.normalize(adj_matrix, p=1, dim=1)
        return neighbors_prob
    
def knn_from_dist(dist_map, k):
    N = dist_map.shape[0]
    adj_matrix = torch.zeros_like(dist_map)

    knn_indices = torch.topk(dist_map, k + 1, largest=False, dim=1).indices[:, 1:]
    batch_indices = torch.arange(N).repeat_interleave(k)
    adj_matrix[batch_indices, knn_indices.reshape(-1)] = 1

    return adj_matrix

class PartialGaussianKernel(GaussianKernel):
    def __init__(self, sigma, mask, type='euclidean'):
        super(PartialGaussianKernel, self).__init__(sigma, type=type)
        self.mask = mask

    def forward(self, features, labels = None, idx = None):
        distances = DistancesKernel(type=self.type)(features)
        distances.masked_fill(self.mask == 0, -float('inf'))
        neighbors_prob = F.softmax(-distances / self.sigma**2, dim=1)
        return neighbors_prob


################## Clustering Kernels ##################
class ClusteringKernel(Kernel):
    def forward(self, cluster_probs, labels=None, idx=None):
        if isinstance(cluster_probs, list) and len(cluster_probs) == 2:
            probs, ema_probs = cluster_probs
        else:
            probs, ema_probs = cluster_probs, cluster_probs
        #probs, ema_probs = F.softmax(probs, dim=-1), F.softmax(ema_probs, dim=-1)

        cluster_sizes = probs.sum(dim=0)
        neighbors_prob = (probs / cluster_sizes) @ ema_probs.t()

        return neighbors_prob
