import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import Accuracy
from metrics import UnsupervisedAccuracy
from loss_helpers import kernel_cross_entropy, kernel_mse_loss, kernel_dot_loss
import copy 
class KernelModel(pl.LightningModule):
    def __init__(self, 
                 mapper, 
                 target_kernel, 
                 learned_kernel, 
                 num_classes=None, 
                 embeddings_map=None, 
                 lr=1e-3, 
                 accuracy_mode=None, 
                 loss_type='cross_entropy', 
                 regularization_coeff=0,
                 use_ema=False,
                 ema_momentum=0.999):
        
        super().__init__()
        self.mapper = mapper
        self.target_kernel = target_kernel
        self.learned_kernel = learned_kernel
        self.embeddings_map = embeddings_map or (lambda x: x)
        self.lr = lr
        self.validation_step_outputs = []
        self.train_loss_epoch = 0
        self.val_loss_epoch = 0
        self.regularization_coeff = regularization_coeff
        self.accuracy_mode = accuracy_mode
        self.use_ema = use_ema
        self.ema_momentum = ema_momentum
        
        # Create EMA mapper if enabled
        if self.use_ema:
            self.ema_mapper = self._create_ema_mapper()
            self._copy_weights_to_ema()
        
        self.loss = self._select_loss_function(loss_type)
        self.train_acc, self.val_acc = self._configure_accuracy_metrics(accuracy_mode, num_classes)

    def _create_ema_mapper(self):
        """Creates a copy of the mapper network for EMA."""
        ema_mapper = copy.deepcopy(self.mapper)
        ema_mapper.eval()  # EMA is not updated through backprop
        for param in ema_mapper.parameters():
            param.requires_grad = False
        return ema_mapper

    def _copy_weights_to_ema(self):
        """Copies the current weights to the EMA mapper."""
        for ema_param, param in zip(self.ema_mapper.parameters(), self.mapper.parameters()):
            ema_param.data.copy_(param.data)

    def _update_ema_weights(self):
        """Updates EMA weights using the momentum parameter."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_mapper.parameters(), self.mapper.parameters()):
                ema_param.data = self.ema_momentum * ema_param.data + (1.0 - self.ema_momentum) * param.data

    def forward(self, x):
        return self.mapper(x)
    
    def forward_ema(self, x):
        """Forward pass through the EMA mapper."""
        return self.ema_mapper(x)
    
    def loss_fn(self, target_kernel, learned_kernel):
        return self.loss(target_kernel, learned_kernel)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def _shared_step(self, batch, batch_idx, phase):
        features, labels, idx = batch
        embeddings = self.embeddings_map(self.forward(features))

        # Generate EMA embeddings if EMA is enabled
        if self.use_ema:
            with torch.no_grad():
                ema_embeddings = self.embeddings_map(self.forward_ema(features))
                ema_embeddings = (ema_embeddings + embeddings) / 2
        else:
            ema_embeddings = embeddings  # Use same embeddings if EMA is disabled

        target_kernel = self.target_kernel(features, labels, idx)
        learned_kernel = self.learned_kernel(([embeddings, ema_embeddings]), labels, idx)

        loss = self.loss_fn(target_kernel, learned_kernel)
        regularization_term = self.regularization_term(embeddings, target_kernel)
        total_loss = loss + self.regularization_coeff * regularization_term

        if phase == 'train':
            self._log_training_step(loss, regularization_term, total_loss)
            self.train_loss_epoch += loss.item()
            self._update_accuracy(embeddings, labels, self.train_acc)
            
            if self.use_ema:
                self._update_ema_weights()  # Update EMA weights during training

        else:
            self.validation_step_outputs.append({
                'embeddings': embeddings,
                'labels': labels,
                'learned_kernel': learned_kernel,
                'target_kernel': target_kernel
            })
            self.log('val_loss', total_loss, on_step=True, on_epoch=False)
            self.val_loss_epoch += total_loss.item()
            self._update_accuracy(embeddings, labels, self.val_acc)

        return loss

    def on_train_epoch_end(self):
        avg_train_loss = self.train_loss_epoch
        self.log('train_loss_epoch', avg_train_loss, on_epoch=True)
        self.train_loss_epoch = 0
        self._compute_and_log_accuracy(self.train_acc, 'train')

    def on_validation_epoch_end(self):
        avg_val_loss = self.val_loss_epoch
        self.log('val_loss_epoch', avg_val_loss, on_epoch=True)
        self.val_loss_epoch = 0
        self._compute_and_log_accuracy(self.val_acc, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

    def regularization_term(self, embeddings, target_kernel):
        return 0  # Default regularization is zero

    def _select_loss_function(self, loss_type):
        if loss_type == 'cross_entropy':
            return kernel_cross_entropy
        elif loss_type == 'mse':
            return kernel_mse_loss
        elif loss_type == 'dot':
            return kernel_dot_loss
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def _configure_accuracy_metrics(self, accuracy_mode, num_classes):
        if accuracy_mode == 'regular':
            return Accuracy(), Accuracy()
        elif accuracy_mode == 'unsupervised':
            return UnsupervisedAccuracy(num_classes), UnsupervisedAccuracy(num_classes)
        else:
            return None, None

    def _update_accuracy(self, embeddings, labels, acc_metric):
        if acc_metric is None:
            return

        if self.accuracy_mode == 'unsupervised':
            clusters = embeddings.argmax(dim=-1)
            acc_metric.update(clusters, labels)
        elif self.accuracy_mode == 'regular':
            predictions = embeddings.argmax(dim=-1)
            acc_metric.update(predictions, labels)

    def _compute_and_log_accuracy(self, acc_metric, phase):
        if acc_metric is None:
            return

        accuracy = acc_metric.compute()
        self.log(f'{phase}_accuracy', accuracy, on_epoch=True)
        print(f'{phase} accuracy: {accuracy}')
        acc_metric.reset()

    def _log_training_step(self, loss, regularization_term, total_loss):
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        self.log('train_reg_loss', regularization_term, on_step=True, on_epoch=False)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=False)

    def _aggregate_validation_outputs(self):
        outputs = self.validation_step_outputs
        all_embeddings = torch.cat([output['embeddings'] for output in outputs], dim=0)
        print(all_embeddings.shape)
        all_labels = torch.cat([output['labels'] for output in outputs], dim=0)
        all_learned_kernels = torch.cat([output['learned_kernel'] for output in outputs], dim=0)
        all_target_kernels = torch.cat([output['target_kernel'] for output in outputs], dim=0)
        self.validation_step_outputs.clear()
        return all_embeddings.cpu(), all_labels.cpu(), all_learned_kernels.cpu(), all_target_kernels.cpu()
    
    