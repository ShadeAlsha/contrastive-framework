import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from metrics import UnsupervisedAccuracy
from loss_helpers import *
import copy
import torch.nn.functional as F


class KernelModel(pl.LightningModule):
    def __init__(self, 
                 mapper, 
                 target_kernel, 
                 learned_kernel,
                 mapper2=None,
                 num_classes=None, 
                 embeddings_map=None,
                 embeddings_map2=None, 
                 lr=1e-3, 
                 accuracy_mode=None, 
                 use_ema=False,
                 ema_momentum=0.999,
                 loss_type='kl', 
                 early_exaggeration=0,
                 decay_factor=0.9,
                 early_exaggeration_epochs=0,
                 linear_probe=False):
        super().__init__()
        self.save_hyperparameters()

        self._init_model_components()
        self._init_training_parameters()
        self._init_loss_function(self.hparams.loss_type)
        self._init_accuracy_metrics(self.hparams.accuracy_mode, self.hparams.num_classes)
        self.acc_logs = {'train': [], 'val': []}

    def _init_model_components(self):
        self.mapper = self.hparams.mapper
        self.target_kernel = self.hparams.target_kernel
        self.learned_kernel = self.hparams.learned_kernel
        self.embeddings_map = self.hparams.embeddings_map or nn.Identity()
        self.embeddings_map2 = self.hparams.embeddings_map2 or self.embeddings_map

        # EMA or alternate mapper initialization
        self.mapper2 = self._initialize_mapper2()

        # Linear probe setup
        self.linear_probe = self.hparams.linear_probe
        self.linear_classifier = (nn.Linear(self.mapper.output_dim, self.hparams.num_classes) 
                                  if self.linear_probe else nn.Identity())

    def _initialize_mapper2(self):
        if self.hparams.use_ema:
            mapper2 = self._create_ema_mapper()
            self._copy_weights_to_ema()
        elif self.hparams.mapper2:
            mapper2 = self.hparams.mapper2
        else:
            mapper2 = self.mapper
        return mapper2

    def _init_training_parameters(self):
        self.lr = self.hparams.lr
        self.early_exaggeration = self.hparams.early_exaggeration
        self.early_exaggeration_epochs = self.hparams.early_exaggeration_epochs
        self.decay_factor = self.hparams.decay_factor

        self.train_loss_epoch = 0
        self.val_loss_epoch = 0
        self.validation_step_outputs = []
        self.training_kernels = []

    def _init_loss_function(self, loss_type):
        loss_functions = {
            'kl': kernel_cross_entropy,
            'l2': kernel_mse_loss,
            'tv': kernel_tv_loss,
            'hellinger': kernel_hellinger_loss,
            'orthogonality_loss': kernel_dot_loss,
            'jsd': jsd_loss,
            'none': lambda x, y: 0.0,
        }
        self.loss = loss_functions.get(loss_type)
        if not self.loss:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def _init_accuracy_metrics(self, accuracy_mode, num_classes):
        accuracy_classes = {
            'regular': lambda: Accuracy(task="multiclass", num_classes=num_classes),
            'unsupervised': lambda: UnsupervisedAccuracy(num_classes),
        }
        self.train_acc, self.val_acc = (accuracy_classes.get(accuracy_mode, lambda: (None, None))() for _ in range(2))

    # === Kernel Computation Methods ===
    def _compute_kernel(self, kernel_fn, features1, features2, labels, idx):
        return kernel_fn([features1, features2], labels, idx)

    def _compute_target_kernel(self, features1, features2, labels, idx):
        return self._compute_kernel(self.target_kernel, features1, features2, labels, idx)

    def _compute_learned_kernel(self, embeddings1, embeddings2, labels, idx):
        return self._compute_kernel(self.learned_kernel, embeddings1, embeddings2, labels, idx)

    # === Main Forward Pass ===
    def forward(self, x1, x2):
        return self.mapper(x1), self.mapper2(x2)

    def get_embeddings(self, embeddings1, embeddings2):
        return self.embeddings_map(embeddings1), self.embeddings_map2(embeddings2)

    # === Shared Training & Validation Steps ===
    def _shared_step(self, batch, phase):
        features, labels, idx = batch
        if features.shape[1] == 2: #TODO: this is a hacky way to handle the case where we have two views, need to fix this
            features1, features2 = features[:, 0, :], features[:, 1, :]
        else:
            features1, features2 = features[:, 0, :], features[:, 0, :]

        inner_embeddings1, inner_embeddings2 = self.forward(features1, features2)
        projection1, projection2 = self.get_embeddings(inner_embeddings1, inner_embeddings2)

        logits = self.linear_classifier(inner_embeddings1.detach())
        learned_kernel = self._compute_learned_kernel(projection1, projection2, labels, idx)
        target_kernel = self._compute_target_kernel(features1, features2, labels, idx)

        kl_loss = self.loss(target_kernel, learned_kernel)
        early_exaggeration_loss = self._compute_early_exaggeration_term(projection1, projection2, target_kernel)
        linear_probe_loss = self._compute_linear_probe_loss(logits, labels)
        total_loss = self._compute_total_loss(kl_loss, early_exaggeration_loss, linear_probe_loss)

        if self.linear_probe:
            logits = F.softmax(logits, dim=-1)
        self._log_step_metrics(phase, total_loss, projection1, logits, labels, learned_kernel, target_kernel)
        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, 'train')
        if self.hparams.use_ema:
            self._update_ema_weights()
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def _compute_total_loss(self, kl_loss, early_exaggeration_loss, linear_probe_loss):
        self.log('kl_loss', kl_loss)
        self.log('early_exaggeration_loss', early_exaggeration_loss)
        self.log('linear_probe_loss', linear_probe_loss)
        return kl_loss + linear_probe_loss + self.hparams.early_exaggeration * early_exaggeration_loss

    def _compute_linear_probe_loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels.long()) if self.linear_probe else 0

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

    # === Optimizer and Scheduler ===
    def configure_optimizers(self): #TODO: update params to accept multiple mappers 
        params = [
            {'params': self.mapper.parameters()},
            {'params': self.embeddings_map.parameters()},
        ]
        if self.linear_probe:
            params.append({'params': self.linear_classifier.parameters(), 'lr': 5e-3})

        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

    # === EMA Methods ===
    def _create_ema_mapper(self):
        ema_mapper = copy.deepcopy(self.mapper)
        ema_mapper.eval()
        for param in ema_mapper.parameters():
            param.requires_grad = False
        return ema_mapper

    def _copy_weights_to_ema(self):
        for ema_param, param in zip(self.mapper2.parameters(), self.mapper.parameters()):
            ema_param.data.copy_(param.data)

    def _update_ema_weights(self):
        with torch.no_grad():
            for ema_param, param in zip(self.mapper2.parameters(), self.mapper.parameters()):
                ema_param.data = self.hparams.ema_momentum * ema_param.data + (1.0 - self.hparams.ema_momentum) * param.data

    # === Accuracy Helpers ===
    def _update_accuracy(self, logits, labels, acc_metric):
        if acc_metric:
            predictions = logits.argmax(dim=-1)
            acc_metric.update(predictions, labels)

    def on_train_epoch_end(self):
        self.log('train_loss_epoch', self.train_loss_epoch, on_epoch=True)
        self.train_loss_epoch = 0
        self._compute_and_log_accuracy(self.train_acc, 'train')

    def on_validation_epoch_end(self):
        self.log('val_loss_epoch', self.val_loss_epoch, on_epoch=True)
        self.val_loss_epoch = 0
        self._compute_and_log_accuracy(self.val_acc, 'val')
        #self._aggregate_validation_outputs()
        self.validation_step_outputs.clear()

    def _compute_and_log_accuracy(self, acc_metric, phase):
        if acc_metric:
            accuracy = acc_metric.compute()
            self.log(f'{phase}_accuracy', accuracy, on_epoch=True)
            self.acc_logs[phase].append(accuracy)
            print(f'max {phase} accuracy: {max(self.acc_logs[phase])}')

    # === Regularization ===
    def _compute_early_exaggeration_term(self, embeddings1, embeddings2, target_kernel):
        if self.current_epoch < self.hparams.early_exaggeration_epochs:
            distance = torch.cdist(embeddings1, embeddings2) * target_kernel
            return distance.sum()
        return 0
