import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_embeddings(embeddings, labels):
    """Plot 2D embeddings with labels.

    Args:
        embeddings (torch.Tensor): The embeddings to plot.
        labels (torch.Tensor): The corresponding labels for the embeddings.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embeddings[:, 0].cpu(), y=embeddings[:, 1].cpu(), hue=labels.cpu(), palette=sns.color_palette("tab10"), edgecolor='none')
    plt.title("2D Embeddings Visualization", fontsize=16)
    plt.xlabel("Embedding Dimension 1", fontsize=14)
    plt.ylabel("Embedding Dimension 2", fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    return plt

def plot_neighborhood_distribution(learned_kernel, target_kernel):
    """Plot the neighborhood distribution of learned and target kernels.

    Args:
        learned_kernel (torch.Tensor): The learned kernel values.
        target_kernel (torch.Tensor): The target kernel values.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    plt.figure(figsize=(10, 5))
    # Example plotting logic (you can customize this)
    plt.plot(learned_kernel.cpu().numpy(), label='Learned Kernel', color='blue')
    plt.plot(target_kernel.cpu().numpy(), label='Target Kernel', color='orange')
    plt.title("Neighborhood Distribution", fontsize=16)
    plt.xlabel("Neighbors", fontsize=14)
    plt.ylabel("Kernel Value", fontsize=14)
    plt.legend()
    plt.grid(True)
    return plt

def plot_cluster_sizes(labels):
    """Plot the distribution of cluster sizes.

    Args:
        labels (torch.Tensor): The labels corresponding to the clusters.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    plt.figure(figsize=(8, 6))
    cluster_sizes = torch.bincount(labels)
    cluster_sizes = cluster_sizes[cluster_sizes.nonzero()].flatten()
    cluster_sizes, _ = torch.sort(cluster_sizes, descending=True)
    
    plt.plot(cluster_sizes.cpu(), color='skyblue')
    plt.title('Cluster Sizes', fontsize=16)
    plt.xlabel('Cluster Index', fontsize=14)
    plt.ylabel('Number of Points', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    return plt
