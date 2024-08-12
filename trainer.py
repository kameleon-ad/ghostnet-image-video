import torch.nn as nn


class Trainer:
    def __init__(self, model: nn.Module, optimizer, criterion, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self._train_one_epoch(epoch)
            self._validate(epoch)

    def _train_one_epoch(self, epoch):
        self.model.train()

    def _validate(self, epoch):
        self.model.eval()
