from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Literal["Adam"],
        lr: float,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        device_ids: list[int],
    ):
        if optimizer == "Adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.model = nn.DataParallel(model, device_ids=device_ids).to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            self._train_one_epoch(epoch)
            self._validate(epoch)

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0

        non_blocking = self.train_loader.pin_memory

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=non_blocking)
            labels = labels.to(self.device, non_blocking=non_blocking)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * labels.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def _validate(self, epoch: int):
        self.model.eval()
        running_loss = 0.0
        correct_count = 0
        total_count = 0

        non_blocking = self.val_loader.pin_memory

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=non_blocking)
                labels = labels.to(self.device, non_blocking=non_blocking)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                predicted = (outputs >= 0.5).float()
                total_count += labels.size(0)
                correct_count += (predicted == labels).sum().item()

        accuracy = 100 * correct_count / total_count
        epoch_loss = running_loss / total_count
        return epoch_loss, accuracy
