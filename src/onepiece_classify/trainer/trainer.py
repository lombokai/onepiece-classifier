from operator import le
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from typing import Dict
from tqdm.auto import tqdm


class Trainer:

    def __init__(
        self,  
        max_epochs: int,
        loss_fn: nn.Module,
        optim: torch.optim.Optimizer,
        learning_rate: float
    ):

        self.max_epochs = max_epochs
        self.loss_fn = loss_fn
        self.optim = optim
        self.learning_rate = learning_rate
        
        self._train_loss = []
        self._train_acc = []
        self._val_loss = []
        self._val_acc = []

    def train_step(self, batch, batch_idx: int = None):
        X, y = batch
        output = self.model(X)

        loss = self.loss_fn(output, y)
        train_loss = loss.item()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        y_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        train_acc = (y_class == y).sum().item()/len(y_class)

        return train_loss, train_acc

    def val_step(self, batch, batch_idx: int=None):
        X, y = batch
        output = self.model(X)

        loss = self.loss_fn(output, y)
        val_loss = loss.item()

        y_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        val_acc = (y_class == y).sum().item()/len(y_class)

        return val_loss, val_acc

    def train_batch(self, epoch: int=None):
        self.model.train()

        self.train_loss = 0
        self.train_acc = 0
        for batch_idx, batch in enumerate(self.loader.train_loader):
            _loss, _acc = self.train_step(batch=batch, batch_idx=batch_idx)
            self.train_loss += _loss
            self.train_acc += _acc

        avg_loss = self.train_loss/len(self.loader.train_loader)
        avg_acc = self.train_acc/len(self.loader.train_loader)
        
        return avg_loss, avg_acc


    def val_batch(self, epoch: int=None):
        self.model.eval()

        self.val_loss = 0
        self.val_acc = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.loader.valid_loader):
                _loss, _acc = self.val_step(batch=batch, batch_idx=batch_idx)
                self.val_loss += _loss
                self.val_acc += _acc

        avg_loss = self.val_loss/len(self.loader.valid_loader)
        avg_acc = self.val_acc/len(self.loader.valid_loader)

        return avg_loss, avg_acc

    def run(self):

        for epoch in tqdm(range(self.max_epochs)):
            train_epoch_loss = self.train_batch(epoch)
            val_epoch_loss = self.val_batch(epoch)

            self._train_loss.append(train_epoch_loss)
            self._val_loss.append(val_epoch_loss)

            self._train_acc.append(train_epoch_loss)
            self._val_acc.append(val_epoch_loss)

    # def save_model(self, path_to_save):
    #     torch.save(self.model.state_dict(), path_to_save)

    def fit(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        self.model = model
        self.loader = loader
        self.run()
        # self.save_model()
