import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from metrics import UnsupervisedAccuracy
from loss_helpers import *
import copy
from kernels import CauchyKernel
class KernelModel(pl.LightningModule):
    def __init__(self, 
                 mapper, 
                 target_kernel, 
                 learned_kernel, 
                 num_classes=None, 
                 embeddings_map=None, 
                 lr=1e-3, 
                 accuracy_mode=None, 
                 use_ema=False,
                 ema_momentum=0.999,
                 loss_type='cross_entropy', 
                 initial_learned_leak=0, 
                 initial_target_leak=0, 
                 initial_beta=0.5,
                 batch_view_coeff=0,
                 early_exaggeration=0,
                 decay_factor=0.99,
                 early_exaggeration_epochs=0,
                 linear_probe=False,
                 bt_leak = 0):
        
        super().__init__()
        self.save_hyperparameters()
        
        self._init_model_components()
        self._init_training_parameters()
        self._init_loss_function(loss_type)
        self._init_accuracy_metrics(accuracy_mode, num_classes)
        self.bt_leak = bt_leak
        print("initialized")
        

    def _init_model_components(self):
        self.mapper = self.hparams.mapper
        self.target_kernel = self.hparams.target_kernel
        self.learned_kernel = self.hparams.learned_kernel
        self.embeddings_map = self.hparams.embeddings_map or nn.Identity()
        self.use_ema = self.hparams.use_ema
        
        if self.use_ema:
            self.ema_mapper = self._create_ema_mapper()
            self._copy_weights_to_ema()
        
        self.linear_probe = self.hparams.linear_probe
        if self.hparams.linear_probe:
            self.linear_classifier = nn.Linear(self.mapper.output_dim, self.hparams.num_classes)
        else:
            self.linear_classifier = nn.Identity()
    def _init_training_parameters(self):
        self.lr = self.hparams.lr
        self.early_exaggeration = self.hparams.early_exaggeration
        self.early_exaggeration_epochs = self.hparams.early_exaggeration_epochs
        self.batch_view_coeff = self.hparams.batch_view_coeff
        self.learned_leak = self.hparams.initial_learned_leak
        self.target_leak = self.hparams.initial_target_leak
        self.beta = self.hparams.initial_beta
        self.decay_factor = self.hparams.decay_factor

        self.train_loss_epoch = 0
        self.val_loss_epoch = 0
        self.validation_step_outputs = []
        self.training_kernels = []

    def _init_loss_function(self, loss_type):
        loss_functions = {
            'cross_entropy': kernel_cross_entropy,
            'l2': kernel_mse_loss,
            'orthogonality_loss': kernel_dot_loss,
            'barlow_twins': barlow_twins,
            'none': lambda x, y: 0.0,
        }
        self.loss = loss_functions.get(loss_type)
        if self.loss is None:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def _init_accuracy_metrics(self, accuracy_mode, num_classes):
        if accuracy_mode == 'regular':
            self.train_acc, self.val_acc = Accuracy(task="multiclass", num_classes=num_classes), Accuracy(task="multiclass", num_classes=num_classes)
        elif accuracy_mode == 'unsupervised':
            self.train_acc, self.val_acc = UnsupervisedAccuracy(num_classes), UnsupervisedAccuracy(num_classes)
        else:
            self.train_acc, self.val_acc = None, None

    def forward(self, x):
        return self.mapper(x)
    
    def forward_ema(self, x):
        return self.ema_mapper(x) if self.use_ema else None

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, 'train')
        self._update_leak_and_beta()
        if self.use_ema:
            self._update_ema_weights()
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')
    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def _shared_step(self, batch, phase):
        features, labels, idx = batch
        features, labels, idx = features.to(self.device), labels.to(self.device), idx.to(self.device)
        inner_embeddings = self.forward(features)
        projection = self.embeddings_map(inner_embeddings)
        logits = self.linear_classifier(inner_embeddings.detach())
        
        ema_projection = self._compute_ema_embeddings(features, projection)
        learned_kernel = self._compute_learned_kernel(projection, ema_projection, labels, idx)
        if phase == 'train':
            target_kernel = self._compute_target_kernel(features, labels, idx)
            kl_loss = self.loss(target_kernel, learned_kernel)
            regularization_loss = self._compute_batch_view_term(projection, target_kernel)
            early_exaggeration_loss = self._compute_early_exaggeration_term(projection, target_kernel)
            linear_probe_loss = self._compute_linear_probe_loss(logits, labels)
            total_loss = self._compute_total_loss(kl_loss, regularization_loss, early_exaggeration_loss, linear_probe_loss)
        else: #loss is zero for validation
            total_loss = torch.tensor(0.0, device=self.device)
            target_kernel = torch.ones_like(learned_kernel)
        self._log_step_metrics(phase, total_loss, projection, logits, labels, learned_kernel, target_kernel)
        return total_loss
    def _compute_linear_probe_loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels.long()) if self.linear_probe else 0
    
    def configure_optimizers(self):
        #use larger learning rate for the linear probe
        optimizer = torch.optim.Adam([
            {'params': self.mapper.parameters()},
            {'params': self.embeddings_map.parameters()},
            {'params': self.linear_classifier.parameters(), 'lr': 5e-3}
        ], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

    # === EMA Methods ===
    def _create_ema_mapper(self):
        ema_mapper = copy.deepcopy(self.mapper)
        ema_mapper.eval()  # EMA is not updated through backprop
        for param in ema_mapper.parameters():
            param.requires_grad = False
        return ema_mapper

    def _copy_weights_to_ema(self):
        for ema_param, param in zip(self.ema_mapper.parameters(), self.mapper.parameters()):
            ema_param.data.copy_(param.data)

    def _update_ema_weights(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_mapper.parameters(), self.mapper.parameters()):
                ema_param.data = self.hparams.ema_momentum * ema_param.data + (1.0 - self.hparams.ema_momentum) * param.data

    def _compute_ema_embeddings(self, features, embeddings):
        if self.use_ema:
            with torch.no_grad():
                ema_embeddings = self.embeddings_map(self.forward_ema(features))
            return self.beta * ema_embeddings + (1 - self.beta) * embeddings
        return embeddings
    def _compute_ema_embeddings(self, features, embeddings):
        if self.use_ema:
            with torch.no_grad():
                ema_embeddings = self.embeddings_map(self.forward_ema(features))
            return self.beta * ema_embeddings + (1 - self.beta) * embeddings
        return embeddings
    # === Kernel Computation Methods ===
    def _compute_target_kernel(self, features, labels, idx):
        return self.target_kernel.leak(self.target_leak)(features, labels, idx)

    def _compute_learned_kernel(self, embeddings, ema_embeddings, labels, idx):
        return self.learned_kernel.leak(self.learned_leak)([embeddings, ema_embeddings], labels, idx)

    # === Training Helpers ===
    def _update_leak_and_beta(self):
        self.learned_leak = max(0, self.learned_leak * self.decay_factor)
        self.target_leak = max(0, self.target_leak * self.decay_factor)
        self.beta = max(0, self.beta * self.decay_factor)

    def _compute_total_loss(self, kl_loss, regularization_loss, early_exaggeration_loss, liner_probe_loss):
        #log the losses
        self.log('kl_loss', kl_loss)
        self.log('regularization_loss', regularization_loss)
        self.log('early_exaggeration_loss', early_exaggeration_loss)
        self.log('linear_probe_loss', liner_probe_loss)
        
        loss = kl_loss + liner_probe_loss
        loss += self.hparams.batch_view_coeff * regularization_loss
        loss += self.hparams.early_exaggeration * early_exaggeration_loss
        return loss

    def _log_step_metrics(self, phase, total_loss, embeddings, logits, labels, learned_kernel, target_kernel):
        if phase == 'train':
            self.train_loss_epoch += total_loss.item()
            self._update_accuracy(logits, labels, self.train_acc)            
        else:
            self.validation_step_outputs.append({
                'learned_kernel': learned_kernel,
                'target_kernel': target_kernel,
                'embeddings': embeddings,
                'logits': logits,
                'labels': labels,
            })
            self.val_loss_epoch += total_loss.item()
            self._update_accuracy(logits, labels, self.val_acc)
    def _aggregate_validation_outputs(self):
        outputs = self.validation_step_outputs
        all_embeddings = torch.cat([output['embeddings'] for output in outputs], dim=0)
        all_logits = torch.cat([output['logits'] for output in outputs], dim=0)
        all_labels = torch.cat([output['labels'] for output in outputs], dim=0)
        all_learned_kernels = torch.cat([output['learned_kernel'] for output in outputs], dim=0)
        all_target_kernels = torch.cat([output['target_kernel'] for output in outputs], dim=0)
        
        # Clear the outputs after processing to free memory
        self.validation_step_outputs.clear()
        
        return all_embeddings.cpu(), all_logits.cpu(), all_labels.cpu(), all_learned_kernels.cpu(), all_target_kernels.cpu()

    
    # === Accuracy Helpers ===
    def _update_accuracy(self, embeddings, labels, acc_metric):
        if acc_metric is None:
            return
        predictions = embeddings.argmax(dim=-1)
        acc_metric.update(predictions, labels)

    def on_train_epoch_end(self):
        self.log('train_loss_epoch', self.train_loss_epoch, on_epoch=True)
        self.train_loss_epoch = 0
        self._compute_and_log_accuracy(self.train_acc, 'train')

    def on_validation_epoch_end(self):
        self.log('val_loss_epoch', self.val_loss_epoch, on_epoch=True)
        self.val_loss_epoch = 0
        self._compute_and_log_accuracy(self.val_acc, 'val')

    def _compute_and_log_accuracy(self, acc_metric, phase):
        if acc_metric:
            accuracy = acc_metric.compute()
            self.log(f'{phase}_accuracy', accuracy, on_epoch=True)
            acc_metric.reset()
            print(f'{phase} accuracy: {accuracy}')

    # === Regularization ===
    def _compute_early_exaggeration_term(self, embeddings, target_kernel):
        if self.current_epoch < self.hparams.early_exaggeration_epochs:
            distance = -(embeddings @ embeddings.T)*target_kernel
            return distance.sum()
        else:
            return 0
    def _compute_batch_view_term(self, embeddings, target_kernel):
        if self.batch_view_coeff > 0:
            return clustering_twins(embeddings, target_kernel, leak = self.bt_leak)
            #return BarlowTwinsLoss(embeddings.device)(embeddings, target_kernel)       
        return 0
    def _cluster_size_regularization(self, embeddings, target_kernel):
        return -torch.log(embeddings.sum(dim=0)).sum()
