
import torch
import torch.nn.functional as F


def kernel_cross_entropy(target_kernel_matrix, learned_kernel_matrix, eps=1e-10):
    target_flat = target_kernel_matrix.flatten()
    learned_flat = learned_kernel_matrix.flatten()

    non_zero_mask = target_flat > 0

    target_filtered = target_flat[non_zero_mask]
    learned_filtered = learned_flat[non_zero_mask]

    cross_entropy_loss = -torch.mean(target_filtered * torch.log(learned_filtered))

    return cross_entropy_loss

def kernel_mse_loss(target_kernel_matrix, learned_kernel_matrix):
    return F.mse_loss(target_kernel_matrix, learned_kernel_matrix).mean()

def kernel_dot_loss(target_kernel_matrix, learned_kernel_matrix):
   return -(target_kernel_matrix*learned_kernel_matrix).mean()