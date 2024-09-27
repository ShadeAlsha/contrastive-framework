
import torch
import torch.nn.functional as F
from kernels import GaussianKernel
from torch.nn import CrossEntropyLoss

def kernel_cross_entropy(target_kernel_matrix, learned_kernel_matrix, eps=1e-10):
    target_flat = target_kernel_matrix.flatten()
    learned_flat = learned_kernel_matrix.flatten()

    non_zero_mask = target_flat > 0

    target_filtered = target_flat[non_zero_mask]
    learned_filtered = learned_flat[non_zero_mask]

    cross_entropy_loss = -torch.sum(target_filtered * torch.log(learned_filtered))
    N = target_kernel_matrix.shape[0]
    return cross_entropy_loss/N

def kernel_mse_loss(target_kernel_matrix, learned_kernel_matrix):
    return ((target_kernel_matrix-learned_kernel_matrix)**2).mean()

def kernel_dot_loss(target_kernel_matrix, learned_kernel_matrix):
   return -(target_kernel_matrix*learned_kernel_matrix).mean()

def clustering_twins(embeddings, target_kernel, leak=0.4):
    n, d = embeddings.size()

    Pxy = target_kernel 
    Exy = (embeddings.t() @ Pxy @ embeddings)
    #Exx = torch.einsum("bi,ci, bc->i", embeddings, embeddings, Pxy) #shape is (d, d)
    reg_x = embeddings.sum(0) #shape is (d,)
    
    C = Exy/reg_x
    target_w_leak = (1-leak)*torch.eye(C.size(0), device=C.device)+leak/d
    return kernel_cross_entropy(target_w_leak, C)

def barlow_twins(embeddings, target_kernel, lambd= 0.001, leak=0.4):
    n, d = embeddings.size()
    #embeddings shape is (n, d)
    #embeddings = (embeddings - embeddings.mean(0))/embeddings.std(0)
    Pxy = target_kernel #shape is (n, n)
    
    #Exy = torch.einsum("bi,cj,bc->ij", embeddings, embeddings, Pxy) #shape is (d, d)
    Exy = embeddings.t() @ Pxy @ embeddings
    reg_x = embeddings.sum(0) #shape is (d,)
    c = Exy/(reg_x)
    c = (c + c.t())/2
    #c = c - ((1-leak)*torch.eye(c.size(0), device=c.device)+leak/d)
    #return 
    on_diag = torch.diagonal(1-c).sum()
    off_diag = off_diagonal(c).sum()
    loss = on_diag + lambd* off_diag
    return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, device= 'cuda', lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, embeddings: torch.Tensor):
        n, d = embeddings.size()
        #embeddings shape is (n, d)
        embeddings = embeddings.reshape(n//2, 2, d)
        z_a = embeddings[:, 0, :]
        z_b = embeddings[:, 1, :]
        
        z_a = (z_a - z_a.mean(0))/z_a.std(0)
        z_b = (z_b - z_b.mean(0))/z_b.std(0)
        
        c = z_a.t() @ z_b / (n//2)
        
        on_diag = torch.diagonal(1-c).pow(2).sum()
        off_diag = off_diagonal(c).pow(2).sum()
        loss = on_diag + self.lambda_param* off_diag
        return loss
def barlow_twins_kl(embeddings, target_kernel, lambd= 0.001, leak=0.4):
    n, d = embeddings.size()
    #embeddings shape is (n, d)
    embeddings = embeddings/(embeddings.norm(dim=0, keepdim=True))
    #embeddings = F.sigmoid(embeddings)
    Pxy = target_kernel #shape is (n, n)
    Exy = embeddings.t() @ Pxy @ embeddings #shape is (d, d)
    
    return CrossEntropyLoss()(Exy, torch.arange(d).to(embeddings.device))

def barlow_twins_pairs_kl(embeddings):
    n, d = embeddings.size()
    #embeddings shape is (n, d)
    #embeddings = embeddings/(embeddings.norm(dim=0, keepdim=True))
    embeddings = embeddings.reshape(n//2, 2, d)
    #embeddings = (embeddings - embeddings.mean(0))/embeddings.std(0)
    z_a = embeddings[:, 0, :] 
    z_b = embeddings[:, 1, :]
    
    #E = - torch.cdist(z_a.T, z_b.T, p=2)/(n//2)
    z_a = (z_a - z_a.mean(0))/z_a.std(0)
    z_b = (z_b - z_b.mean(0))/z_b.std(0)
    E = (z_a.T@z_b)/(n//2)
    
    #log_probs = torch.log_softmax(Exy.diagonal(), dim=-1) #shape is (d, d)
    #indep_kernel = torch.eye(d, device=embeddings.device)
    l1 = CrossEntropyLoss()(E, torch.arange(d).to(embeddings.device))
    l2 = CrossEntropyLoss()(E.T, torch.arange(d).to(embeddings.device))
    return l1+l2