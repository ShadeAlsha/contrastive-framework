import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

class PlotLogger(pl.Callback):
    def __init__(self, show_plots=False, selected_plots=None):
        super().__init__()
        self.show_plots = show_plots
        # List of plot functions the user wants to show
        self.selected_plots = selected_plots if selected_plots is not None else ['neighborhood_dist', 'embeddings']

    def plot_embeddings(self, embeddings, labels, ax=None):
        """Plot 2D embeddings with labels on a specific axis."""
        if ax is None:
            ax = plt.gca()  # Get current axis
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels,
                        palette=sns.color_palette("tab10"), edgecolor='none', ax=ax)
        ax.set_title("2D Embeddings Visualization", fontsize=16)
        ax.set_xlabel("Embedding Dimension 1", fontsize=14)
        ax.set_ylabel("Embedding Dimension 2", fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5)

    def plot_neighborhood_dist(self, learned_kernel, target_kernel, ax=None):
        """Plot neighborhood distribution using learned and target kernels."""
        #normalize the learned and target kernels
        learned_kernel = F.normalize(learned_kernel, p=1, dim=-1)
        target_kernel = F.normalize(target_kernel, p=1, dim=-1)
        
        if ax is None:
            ax = plt.gca()  # Get current axis
        _, indices = torch.sort(target_kernel, dim=-1)
        probs_sorted = torch.gather(learned_kernel, 1, indices)
        t_sorted = torch.gather(target_kernel, 1, indices)

        probs_avg, probs_std = probs_sorted.mean(dim=0), probs_sorted.std(dim=0)
        t_probs_avg, t_probs_std = t_sorted.mean(dim=0), t_sorted.std(dim=0)

        x_values = list(reversed(range(probs_avg.size(0))))
        probs_avg, probs_std = probs_avg.flip(0), probs_std.flip(0)
        t_probs_avg, t_probs_std = t_probs_avg.flip(0), t_probs_std.flip(0)

        ax.plot(x_values, t_probs_avg.numpy(), label='Target Distribution $P_i$', color='orange', linewidth=2)
        ax.plot(x_values, probs_avg.numpy(), label='Learned Distribution $Q_i$', linewidth=2)
        ax.fill_between(x_values, (probs_avg - probs_std).numpy(), (probs_avg + probs_std).numpy(), alpha=0.3)
        ax.fill_between(x_values, (t_probs_avg - t_probs_std).numpy(), (t_probs_avg + t_probs_std).numpy(),
                        color='orange', alpha=0.3)

        ax.set_yscale('log')
        ax.set_xlabel('Neighbors Ordered by Proximity', fontsize=14)
        ax.set_ylabel('Selection Probability', fontsize=14)
        ax.set_title('Neighbor Selection Probability Distributions', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    def plot_cluster_sizes(self, probablities, ax=None):
        """Plot the sizes of the clusters."""
        if ax is None:
            ax = plt.gca()  # Get current axis
        cluster_sizes = probablities.sum(dim=0)
        cluster_sizes, _ = torch.sort(cluster_sizes, descending=True)

        ax.plot(cluster_sizes.cpu(), color='skyblue')
        ax.set_title('Cluster Sizes', fontsize=16)
        ax.set_xlabel('Cluster Index', fontsize=14)
        ax.set_ylabel('Number of Points', fontsize=14)
        ax.grid(True, linestyle='--', linewidth=0.5)

    def plot_probabilities_star(self, probs, labels, cluster_names=None, radius=1, ax=None):
        """Plot star-shaped probability visualization."""
        if cluster_names is None:
            cluster_names = [' ']*(probs.shape[1])

        if ax is None:
            ax = plt.gca()

        n_points, n_clusters = probs.shape
        theta = torch.linspace(0, 2 * torch.pi, n_clusters + 1, dtype=probs.dtype)[:-1]
        vertices_x = radius * torch.cos(theta)
        vertices_y = radius * torch.sin(theta)

        x = probs @ vertices_x
        y = probs @ vertices_y

        cmap = plt.get_cmap('tab10')
        colors = cmap(torch.linspace(0, 1, 10))[:n_clusters]

        ax.scatter(vertices_x, vertices_y, c='black', s=300, marker='*', alpha=0.5)

        for i in range(n_clusters):
            mask = (labels == i)
            ax.scatter(x[mask], y[mask], color=colors[i], alpha=0.8, label=cluster_names[i])

        for i, name in enumerate(cluster_names):
            ax.text(vertices_x[i] * 1.05, vertices_y[i] * 1.05, f' {name}', ha='center', fontsize=14, color='black')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    def log_figure_to_tensorboard(self, figure, tag, trainer, pl_module):
        """Log a matplotlib figure to TensorBoard."""
        buf = BytesIO()
        figure.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        image = np.array(image)
        trainer.logger.experiment.add_image(tag, image, global_step=trainer.global_step, dataformats='HWC')
        plt.close(figure)

    def show_selected_plots(self, trainer, pl_module, embeddings, labels, learned_kernel, target_kernel, probs=None):
        """Show or log the selected plots."""
        fig, axes = plt.subplots(1, len(self.selected_plots), figsize=(6 * len(self.selected_plots), 5))

        if len(self.selected_plots) == 1:
            axes = [axes]

        for ax, plot_name in zip(axes, self.selected_plots):
            if plot_name == 'embeddings':
                self.plot_embeddings(embeddings, labels, ax=ax)
            elif plot_name == 'neighborhood_dist':
                self.plot_neighborhood_dist(learned_kernel, target_kernel, ax=ax)
            elif plot_name == 'probabilities_star':
                self.plot_probabilities_star(embeddings, labels, ax=ax)
            elif plot_name == 'cluster_sizes':
                self.plot_cluster_sizes(embeddings, ax=ax)

        if self.show_plots:
            plt.tight_layout()
            plt.show()
        else:
            for i, plot_name in enumerate(self.selected_plots):
                self.log_figure_to_tensorboard(fig, plot_name, trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log or display plots at the end of each validation epoch."""
        all_embeddings, all_labels, all_learned_kernels, all_target_kernels = pl_module._aggregate_validation_outputs()
        self.show_selected_plots(trainer, pl_module, all_embeddings, all_labels, all_learned_kernels, all_target_kernels)
