import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F

class PlotLogger:
    def __init__(self, show_plots=False, selected_plots=None, tensorboard_writer=None):
        self.show_plots = show_plots
        self.tensorboard_writer = tensorboard_writer  # TensorBoard writer, if logging is required
        self.selected_plots = selected_plots if selected_plots is not None else ['neighborhood_dist', 'embeddings']

    def plot_embeddings(self, embeddings, labels, ax=None):
        if ax is None:
            ax = plt.gca()
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels,
                        palette=sns.color_palette("tab10"), edgecolor='none', ax=ax)
        ax.set_title("2D Embeddings Visualization", fontsize=16)
        ax.set_xlabel("Embedding Dimension 1", fontsize=14)
        ax.set_ylabel("Embedding Dimension 2", fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5)

    def plot_neighborhood_dist(self, learned_kernel, target_kernel, ax=None):
        learned_kernel = F.normalize(learned_kernel, p=1, dim=-1)
        target_kernel = F.normalize(target_kernel, p=1, dim=-1)
        
        if ax is None:
            ax = plt.gca()
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

    def log_figure_to_tensorboard(self, figure, tag, global_step):
        buf = BytesIO()
        figure.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        image = np.array(image)
        self.tensorboard_writer.add_image(tag, image, global_step=global_step, dataformats='HWC')
        plt.close(figure)

    def show_selected_plots(self, embeddings, labels, learned_kernel, target_kernel, epoch):
        fig, axes = plt.subplots(1, len(self.selected_plots), figsize=(6 * len(self.selected_plots), 5))

        if len(self.selected_plots) == 1:
            axes = [axes]

        for ax, plot_name in zip(axes, self.selected_plots):
            if plot_name == 'embeddings':
                self.plot_embeddings(embeddings, labels, ax=ax)
            elif plot_name == 'neighborhood_dist':
                self.plot_neighborhood_dist(learned_kernel, target_kernel, ax=ax)

        if self.show_plots:
            plt.tight_layout()
            plt.show()
        else:
            for i, plot_name in enumerate(self.selected_plots):
                if self.tensorboard_writer:
                    self.log_figure_to_tensorboard(fig, f"{plot_name}/epoch_{epoch}", epoch)

    def log_and_plot(self, embeddings, labels, learned_kernel, target_kernel, epoch):
        """Aggregate and plot data at the end of an epoch."""
        self.show_selected_plots(embeddings, labels, learned_kernel, target_kernel, epoch)
