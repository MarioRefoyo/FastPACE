from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Learner(ABC):
    def __init__(self, model: nn.Module,
                 num_epochs,
                 es_patience: int = 30,
                 include_labels_in_forward: bool = False) -> None:
        self.model = model
        self.num_epochs = num_epochs

        self.es_patience = es_patience

        # to be filled by the fit function
        self.is_trained = False
        self.test_results: dict = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.include_labels_in_forward = include_labels_in_forward

    @staticmethod
    def _get_batch_size(batch_size, dataset_len):
        if batch_size == "all_dataset":
            new_batch_size = dataset_len
        elif isinstance(batch_size, int):
            new_batch_size = min(batch_size, dataset_len)
        else:
            raise ValueError("Not valid batch size")
        return new_batch_size

    def fit(self, x, y, optimizer, scheduler, batch_size: int = 64, val_size: float = 0.2, train_loader=None, val_loader=None, verbose=True):

        # self.train_loss: list = []
        # self.val_loss: list = []

        if (train_loader is None) and (val_loader is None):
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=max(val_size, len(np.unique(y)) / len(x)), stratify=y
            )

            batch_size_train = self._get_batch_size(batch_size, x_train.shape[0])
            batch_size_val = self._get_batch_size(batch_size, x_val.shape[0])

            # Create loader
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()),
                batch_size=batch_size_train,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()),
                batch_size=batch_size_val,
                shuffle=False
            )

        # Train model
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        epoch_metrics = []
        best_val_loss = np.inf
        patience_counter = 0
        best_state_dict = None

        # Model training
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            train_loss, train_acc, val_loss, val_acc = self.fit_epoch_specific(
                train_loader, val_loader, optimizer, scheduler, epoch)

            if verbose:
                print(f'Epoch: {epoch + 1}, '
                      f'Train loss: {round(train_loss, 3)}, '
                      f'Train acc: {round(train_acc, 3)}, || '
                      f'Val loss: {round(val_loss, 3)}, '
                      f'Val acc: {round(val_acc, 3)}')
            metrics = {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
            epoch_metrics.append(metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = self.model.state_dict()
                self.best_state_dict = best_state_dict
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter == self.es_patience:
                    if best_state_dict is not None:
                        self.model.load_state_dict(best_state_dict)
                    print('Early stopping!')
                    break

        self.is_trained = True
        self.end_of_training_callback()
        return epoch_metrics

    @abstractmethod
    def fit_epoch_specific(self, train_loader, val_loader, optimizer, scheduler, epoch, **kwargs):
        pass

    @abstractmethod
    def calculate_loss(self, output, y_true):
        pass

    @abstractmethod
    def end_of_training_callback(self):
        pass

    def predict(self, x_test, batch_size: int = 64):

        batch_size_test = self._get_batch_size(batch_size, x_test.shape[0])
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(x_test).float()),
            batch_size=batch_size_test,
            shuffle=False,
        )

        self.model.eval()
        preds_list = []
        for x_t, _ in test_loader:
            x_t = x_t.to(self.device)
            with torch.no_grad():
                preds = self.model.predict(x_t)
                if preds.shape[1] == 1:
                    preds = torch.sigmoid(preds)
                else:
                    preds = torch.softmax(preds, dim=-1)
                if self.device.type == "cuda":
                    preds_list.append(preds.cpu().detach().numpy())
                else:
                    preds_list.append(preds.detach().numpy())
        preds_np = np.concatenate(preds_list)
        return preds_np


class BasicLearner(Learner):
    def __init__(self, model: nn.Module,
                 criterion,
                 num_epochs,
                 es_patience: int = 30,
                 include_labels_in_forward: bool = False) -> None:
        super().__init__(model, num_epochs, es_patience, include_labels_in_forward)
        self.criterion = criterion.to(self.device)

    def calculate_loss(self, output, y_true):
        loss = self.criterion(output, y_true)
        return loss

    def fit_epoch_specific(self, train_loader, val_loader, optimizer, scheduler, epoch, **kwargs):
        n_examples_train = 0
        n_correct_train = 0
        n_examples_val = 0
        n_correct_val = 0

        # Model train
        epoch_train_loss = []
        self.model.train()
        for data in train_loader:
            x_t, y_t = data[0].to(self.device), data[1].to(self.device)
            optimizer.zero_grad()
            if self.include_labels_in_forward:
                output = self.model(x_t, y_t)
            else:
                output = self.model(x_t)
            # Extract predictions from output (IT IS MANDATORY THAT IF OUTPUT IS A TUPLE, FIRST VALUE IS TRULY OUTPUT)
            train_loss = self.calculate_loss(output, y_t)
            if isinstance(output, tuple):
                _, predicted = torch.max(output[0].data, 1)
            else:
                _, predicted = torch.max(output.data, 1)
            n_examples_train += y_t.size(0)
            n_correct_train += (predicted == y_t).sum().item()

            epoch_train_loss.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
        train_loss = np.mean(epoch_train_loss)
        train_acc = n_correct_train / n_examples_train

        # Model validation
        epoch_val_loss = []
        self.model.eval()
        with torch.no_grad():
            for data in val_loader:
                x_v, y_v = data[0].to(self.device), data[1].to(self.device)
                with torch.no_grad():
                    output = self.model.predict(x_v)
                    # Extract predictions from output (IT IS MANDATORY THAT IF OUTPUT IS A TUPLE, FIRST VALUE IS TRULY OUTPUT)
                    val_loss = self.calculate_loss(output, y_v)
                    if isinstance(output, tuple):
                        _, predicted = torch.max(output[0].data, 1)
                    else:
                        _, predicted = torch.max(output.data, 1)
                    epoch_val_loss.append(val_loss.item())
                    n_examples_val += y_v.size(0)
                    n_correct_val += (predicted == y_v).sum().item()
            val_loss = np.mean(epoch_val_loss)
            val_acc = n_correct_val / n_examples_val
            scheduler.step(val_loss)

        return train_loss, train_acc, val_loss, val_acc

    def end_of_training_callback(self):
        pass


