import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import Accuracy 
from metrics import UnsupervisedAccuracy

class KernelModel(pl.LightningModule):
    def __init__(self, 
                 mapper, 
                 target_kernel, 
                 learned_kernel, 
                 num_classes= None, 
                 embeddings_map=None, 
                 lr=1e-3, 
                 accuracy_mode=None, 
                 plot_embeddings = False):
        """
        accuracy_mode: str, options are 'regular', 'unsupervised', or None
        """
        super().__init__()
        self.mapper = mapper
        self.target_kernel = target_kernel
        self.learned_kernel = learned_kernel
        self.embeddings_map = embeddings_map if embeddings_map is not None else lambda x: x
        self.lr = lr
        self.validation_step_outputs = []
        self.train_loss_epoch = 0
        self.val_loss_epoch = 0
        self.plot_embeddings = plot_embeddings

        # Set the accuracy mode: 'regular', 'unsupervised', or None
        self.accuracy_mode = accuracy_mode
        if accuracy_mode == 'regular':
            self.train_acc = Accuracy()
            self.val_acc = Accuracy()
        elif accuracy_mode == 'unsupervised':
            self.train_acc = UnsupervisedAccuracy(num_classes)
            self.val_acc = UnsupervisedAccuracy(num_classes)
        else:
            self.train_acc = None
            self.val_acc = None

    def forward(self, x):
        return self.mapper(x)

    def loss_fn(self, target_kernel, learned_kernel):
        return kernel_cross_entropy(target_kernel, learned_kernel)

    def update_accuracy(self, embeddings, labels, acc_metric):
        """
        Update the accuracy metric based on the accuracy mode.
        """
        if acc_metric is None:
            return  # Skip accuracy calculation if mode is None
        
        if self.accuracy_mode == 'unsupervised':
            clusters = embeddings.argmax(dim=-1)
            acc_metric.update(clusters, labels)
        elif self.accuracy_mode == 'regular':
            predictions = embeddings.argmax(dim=-1)  # Assuming we're classifying based on argmax over embeddings
            acc_metric.update(predictions, labels)

    def compute_and_log_accuracy(self, acc_metric, phase='train'):
        """
        Compute and log the accuracy for either training or validation.
        """
        if acc_metric is None:
            return  # Skip logging accuracy if mode is None

        accuracy = acc_metric.compute()
        self.log(f'{phase}_accuracy', accuracy, on_epoch=True)
        print(f'{phase}_accuracy', accuracy)

        acc_metric.reset()

    def training_step(self, batch, batch_idx):
        features, labels, idx = batch
        embeddings = self.embeddings_map(self.forward(features))
        
        target_kernel = self.target_kernel(features, labels, idx)
        learned_kernel = self.learned_kernel(embeddings, labels, idx)
        
        loss = self.loss_fn(target_kernel, learned_kernel)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        self.train_loss_epoch += loss.item()

        # Update the training accuracy
        self.update_accuracy(embeddings, labels, self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels, idx = batch
        embeddings = self.embeddings_map(self.forward(features))
        
        target_kernel = self.target_kernel(features, labels, idx)
        learned_kernel = self.learned_kernel(embeddings, labels, idx)
        
        loss = self.loss_fn(target_kernel, learned_kernel)

        self.validation_step_outputs.append({
            'embeddings': embeddings,
            'labels': labels,
            'learned_kernel': learned_kernel,
            'target_kernel': target_kernel
        })
        
        self.log('val_loss', loss, on_step=True, on_epoch=False)
        self.val_loss_epoch += loss.item()  # Accumulate validation loss for epoch

        # Update the validation accuracy
        self.update_accuracy(embeddings, labels, self.val_acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]
        
    def on_train_epoch_end(self):
        # Log the total epoch loss for training
        avg_train_loss = self.train_loss_epoch
        self.log('train_loss_epoch', avg_train_loss, on_epoch=True)
        self.train_loss_epoch = 0  # Reset for next epoch

        # Compute and log the training accuracy
        self.compute_and_log_accuracy(self.train_acc, phase='train')

    def on_validation_epoch_end(self):
        # Aggregate embeddings
        all_embeddings, all_labels, all_learned_kernels, all_target_kernels = self.aggregate_validation_embeddings()
        
        if self.plot_embeddings:
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            self.plot_neighborhood_dist(all_learned_kernels, all_target_kernels, limit=0, ax=axs[0])
            if len(all_embeddings[0]) <= 2:
                self.plot_low_dim_embeddings(all_embeddings, all_labels, ax=axs[1])
            else:
                plot_probabilities_star(all_embeddings, all_labels, cluster_names=None, radius = 1, ax=axs[1])
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            self.plot_neighborhood_dist(all_learned_kernels, all_target_kernels, limit=0, ax=None)
            plt.show()
            plt.close()


        # Log the total epoch loss for validation
        avg_val_loss = self.val_loss_epoch
        self.log('val_loss_epoch', avg_val_loss, on_epoch=True)
        self.val_loss_epoch = 0 
        
        # Compute and log the validation accuracy
        self.compute_and_log_accuracy(self.val_acc, phase='val')
    
    def aggregate_validation_embeddings(self):
        outputs = self.validation_step_outputs
        all_embeddings = torch.cat([output['embeddings'] for output in outputs], dim=0)
        all_labels = torch.cat([output['labels'] for output in outputs], dim=0)
        all_learned_kernels = torch.cat([output['learned_kernel'] for output in outputs], dim=0)
        all_target_kernels = torch.cat([output['target_kernel'] for output in outputs], dim=0)
        
        # Clear stored validation step outputs to free memory
        self.validation_step_outputs.clear()
        return all_embeddings.cpu(), all_labels.cpu(), all_learned_kernels.cpu(), all_target_kernels.cpu()
        
    @staticmethod
    def plot_neighborhood_dist(learned_kernel, target_kernel, limit=0, ax=None):
        _, indices = torch.sort(target_kernel, dim=-1)
        probs_sorted = torch.gather(learned_kernel, 1, indices)
        t_sorted = torch.gather(target_kernel, 1, indices)

        probs_avg = probs_sorted.mean(dim=0)
        probs_std = probs_sorted.std(dim=0)
        t_probs_avg = t_sorted.mean(dim=0)
        t_probs_std = t_sorted.std(dim=0)

        x_values = range(probs_avg.size(0))[-limit:]
        
        # Reverse the order for closer to further points
        x_values = list(reversed(x_values))
        probs_avg, probs_std = probs_avg.flip(0), probs_std.flip(0)
        t_probs_avg, t_probs_std = t_probs_avg.flip(0), t_probs_std.flip(0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(x_values, t_probs_avg[-limit:].numpy(), label='Target Probability Distribution $P_i$', color='orange', linewidth=2)
        ax.plot(x_values, probs_avg[-limit:].numpy(), label='Learned Probability Distribution $Q_i$', linewidth=2)
        ax.fill_between(x_values, (probs_avg - probs_std)[-limit:].numpy(), 
                        (probs_avg + probs_std)[-limit:].numpy(), alpha=0.3)
        ax.fill_between(x_values, (t_probs_avg - t_probs_std)[-limit:].numpy(), 
                        (t_probs_avg + t_probs_std)[-limit:].numpy(), color='orange', alpha=0.3)

        ax.set_yscale('log')
        ax.set_xlabel('Neighbors Ordered by Proximity', fontsize=14)
        ax.set_ylabel('Selection Probability', fontsize=14)
        ax.set_title('Neighbor Selection Probability Distributions', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

    @staticmethod
    def plot_low_dim_embeddings(embeddings, labels, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.scatterplot(
            x=embeddings[:, 0], y=embeddings[:, 1], hue=labels, palette=sns.color_palette("tab10"), 
            legend="full", edgecolor='none', ax=ax
        )
        ax.set_title("2D Embeddings Visualization of MNIST", fontsize=16)
        ax.set_xlabel("Embedding Dimension 1", fontsize=14)
        ax.set_ylabel("Embedding Dimension 2", fontsize=14)
        ax.legend(loc='upper right', fontsize=14)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
    


def kernel_cross_entropy(target_kernel_matrix, learned_kernel_matrix, eps=1e-10):
    target_flat = target_kernel_matrix.flatten()
    learned_flat = learned_kernel_matrix.flatten()

    non_zero_mask = target_flat > 0

    target_filtered = target_flat[non_zero_mask]
    learned_filtered = learned_flat[non_zero_mask]

    cross_entropy_loss = -torch.sum(target_filtered * torch.log(learned_filtered))

    return cross_entropy_loss.mean()

def plot_probabilities_star(probs, labels, cluster_names=None, radius=1, ax=None):
    if cluster_names is None:
        cluster_names = [f'Cluster {i}' for i in range(probs.shape[1])]

    n_points, n_clusters = probs.shape

    # Create angles for vertices
    theta = torch.linspace(0, 2 * torch.pi, n_clusters + 1, dtype=probs.dtype)[:-1]  # Match dtype with probs
    vertices_x = radius * torch.cos(theta)
    vertices_y = radius * torch.sin(theta)

    # Compute the x and y coordinates of the points based on probabilities
    x = probs @ vertices_x
    y = probs @ vertices_y

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.get_cmap('tab10')
    colors = cmap(torch.linspace(0, 1, 10))[:n_clusters]  # Clip to n_clusters

    # Plot star points (vertices of the clusters)
    ax.scatter(vertices_x, vertices_y, c='black', s=300, marker='*', alpha=0.5)

    # Plot data points for each cluster
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(x[mask], y[mask], color=colors[i], alpha=0.8, label=cluster_names[i])

    # Write cluster names on each vertex
    for i, name in enumerate(cluster_names):
        ax.text(vertices_x[i] * 1.05, vertices_y[i] * 1.05, ' ' + name, ha='center', fontsize=14, color='black')

    # Set axis ticks and legend
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
