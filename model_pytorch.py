import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from metrics import UnsupervisedAccuracy
from loss_helpers import *
from tqdm import tqdm
import copy

class KernelModel(nn.Module):
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
                 initial_beta=0,
                 batch_view_coeff=0,
                 early_exaggeration=1,
                 decay_factor=0.99,
                 early_exaggeration_epochs=0,
                 linear_probe=False,
                 device='cpu'):
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.use_ema = use_ema
        self.ema_momentum = ema_momentum
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.batch_view_coeff = batch_view_coeff
        self.learned_leak = initial_learned_leak
        self.target_leak = initial_target_leak
        self.beta = initial_beta
        self.decay_factor = decay_factor
        self.linear_probe = linear_probe

        self._initialize_modules(mapper, target_kernel, learned_kernel, num_classes, embeddings_map)
        self._initialize_metrics(accuracy_mode, num_classes)
        self._initialize_loss_function(loss_type)
        self._initialize_ema()

        self.train_loss_epoch = 0
        self.val_loss_epoch = 0
        self.validation_step_outputs = []
        self.training_kernels = []

    def _initialize_modules(self, mapper, target_kernel, learned_kernel, num_classes, embeddings_map):
        self.mapper = mapper.to(self.device)
        self.target_kernel = target_kernel
        self.learned_kernel = learned_kernel
        self.embeddings_map = embeddings_map.to(self.device) if embeddings_map else nn.Identity().to(self.device)

        if self.linear_probe:
            self.linear_classifier = nn.Linear(mapper.output_dim, num_classes).to(self.device)
        else:
            self.linear_classifier = nn.Identity().to(self.device)

    def _initialize_metrics(self, accuracy_mode, num_classes):
        if accuracy_mode == 'regular':
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        elif accuracy_mode == 'unsupervised':
            self.train_acc = UnsupervisedAccuracy(num_classes).to(self.device)
            self.val_acc = UnsupervisedAccuracy(num_classes).to(self.device)
        else:
            self.train_acc = self.val_acc = None

    def _initialize_loss_function(self, loss_type):
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

    def _initialize_ema(self):
        if self.use_ema:
            self.ema_mapper = self._create_ema_mapper()
            self._copy_weights_to_ema()
    ###### Pytorch Lightning methods ######
    def forward(self, x):
        return self.mapper(x)
    
    def forward_ema(self, x):
        return self.ema_mapper(x) if self.use_ema else None

    def training_step(self, batch):
        loss = self._shared_step(batch, 'train')
        self._update_leak_and_beta()
        if self.use_ema:
            self._update_ema_weights()
        return loss

    def validation_step(self, batch):
        return self._shared_step(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.mapper.parameters()},
            {'params': self.embeddings_map.parameters()},
            {'params': self.linear_classifier.parameters(), 'lr': 1e-2}
        ], lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return optimizer, scheduler

    def on_train_epoch_end(self):
        print(f'Train loss: {self.train_loss_epoch}')
        self.train_loss_epoch = 0
        self._compute_and_log_accuracy(self.train_acc, 'train')

    def on_validation_epoch_end(self):
        print(f'Validation loss: {self.val_loss_epoch}')
        self.val_loss_epoch = 0
        self._compute_and_log_accuracy(self.val_acc, 'val')

    ###### forward pass helpers ######
    def _shared_step(self, batch, phase):
        features, labels, idx = self._move_batch_to_device(batch)
        inner_embeddings, projection, ema_projection = self._compute_embeddings(features)
        
        target_kernel = self._compute_target_kernel(features, labels, idx)
        learned_kernel = self._compute_learned_kernel(projection, ema_projection, labels, idx)
        
        total_loss = self._compute_loss(inner_embeddings, labels, projection, target_kernel, learned_kernel)

        logits = self.linear_classifier(inner_embeddings)

        self._log_step_metrics(phase, total_loss, projection, logits, labels, learned_kernel, target_kernel)
        
        return total_loss

    def _compute_embeddings(self, features):
        inner_embeddings = self.forward(features)
        projection = self.embeddings_map(inner_embeddings)
        ema_projection = self._compute_ema_embeddings(features, projection)
        return inner_embeddings, projection, ema_projection
    def _compute_target_kernel(self, features, labels, idx):
        return self.target_kernel.leak(self.target_leak)(features, labels, idx)

    def _compute_learned_kernel(self, embeddings, ema_embeddings, labels, idx):
        return self.learned_kernel.leak(self.learned_leak)([embeddings, ema_embeddings], labels, idx)


    ###### Loss computation ######
    def _compute_loss(self, inner_embeddings, labels, projection, target_kernel, learned_kernel):
        kl_loss = self.loss(target_kernel, learned_kernel)
        regularization_loss = self._batch_view_term(projection, target_kernel)
        early_exaggeration_loss = self._early_exaggeration_term(projection, target_kernel)
        linear_probe_loss = self._compute_linear_probe_loss(inner_embeddings.detach(), labels)
        
        total_loss = kl_loss + linear_probe_loss
        total_loss += self.batch_view_coeff * regularization_loss
        total_loss += self.early_exaggeration * early_exaggeration_loss
        
        return total_loss

    def _early_exaggeration_term(self, embeddings, target_kernel):
        if self.current_epoch < self.early_exaggeration_epochs:
            distance = -(embeddings @ embeddings.T) * target_kernel
            return -distance.sum()
        return 0

    def _batch_view_term(self, embeddings, target_kernel):
        if self.batch_view_coeff > 0:
            #return BarlowTwinsLoss(device=self.device)(embeddings)
            return barlow_twins_pairs_kl(embeddings)
        return 0
    def _compute_linear_probe_loss(self, inner_embeddings, labels):
        if self.linear_probe:
            logits = self.linear_classifier(inner_embeddings.detach())
            return nn.CrossEntropyLoss()(logits, labels)
        else:
            return 0

    ###### EMA helpers ######
    def _compute_ema_embeddings(self, features, embeddings):
        if self.use_ema:
            with torch.no_grad():
                ema_embeddings = self.embeddings_map(self.forward_ema(features))
            return self.beta * ema_embeddings + (1 - self.beta) * embeddings
        return embeddings

    def _create_ema_mapper(self):
        ema_mapper = copy.deepcopy(self.mapper).to(self.device)
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
                ema_param.data = self.ema_momentum * ema_param.data + (1.0 - self.ema_momentum) * param.data

    ###### scheduling and logging helpers ######
    def _update_leak_and_beta(self):
        self.learned_leak = max(0, self.learned_leak * self.decay_factor)
        self.target_leak = max(0, self.target_leak * self.decay_factor)
        self.beta = max(0, self.beta * self.decay_factor)

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
            
    def _compute_and_log_accuracy(self, acc_metric, phase):
        if acc_metric:
            accuracy = acc_metric.compute()
            print(f'{phase} accuracy: {accuracy}')
            acc_metric.reset()
            
    def _update_accuracy(self, embeddings, labels, acc_metric):
        if acc_metric is None:
            return
        predictions = embeddings.argmax(dim=-1)
        acc_metric.update(predictions, labels)

    def _aggregate_validation_outputs(self):
        outputs = self.validation_step_outputs
        all_embeddings = torch.cat([output['embeddings'] for output in outputs], dim=0)
        all_labels = torch.cat([output['labels'] for output in outputs], dim=0)
        all_learned_kernels = torch.cat([output['learned_kernel'] for output in outputs[:-1]], dim=0)
        all_target_kernels = torch.cat([output['target_kernel'] for output in outputs[:-1]], dim=0)
        
        self.validation_step_outputs.clear()
        
        return all_embeddings.cpu(), all_labels.cpu(), all_learned_kernels.cpu(), all_target_kernels.cpu()
    
    def _move_batch_to_device(self, batch):
        features, labels, idx = batch
        return features.to(self.device), labels.to(self.device), idx.to(self.device)

    
# Training loop
def train_model(model, train_loader, val_loader, device, epochs, plot_logger= None):
    model.to(device)
    optimizer, scheduler = model.configure_optimizers()

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        model.current_epoch = epoch
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.on_train_epoch_end()

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                model.validation_step(batch)
        model.on_validation_epoch_end()
        if plot_logger:
            # Extract data for plotting
            all_embeddings, all_labels, all_learned_kernels, all_target_kernels = model._aggregate_validation_outputs()

            # Log and plot
            plot_logger.log_and_plot(all_embeddings, all_labels, all_learned_kernels, all_target_kernels, epoch)
