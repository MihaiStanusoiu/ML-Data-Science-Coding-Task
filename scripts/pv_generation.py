import copy

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from models.pv_forecast_model import PVForecastModel
from scripts.datasets import PVDataset


class Trainer:
    def __init__(self, train_dataset: PVDataset, val_dataset: PVDataset, test_dataset: PVDataset, batch_size: int, seq_len: int, hidden_size: int, num_features: int, num_layers: int, max_epochs: int, early_stopping: int, device: str, model_path: str = None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.device = device

        model_factory = lambda : PVForecastModel(input_size=num_features, hidden_size=hidden_size, nr_layers=num_layers,
                                             output_size=1)

        if model_path:
            try:
                self.model = torch.load(model_path)
                self.model.to(self.device)
            except Exception:
                self.model = model_factory()
        else:
            self.model = model_factory()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        self.validation_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.loss_fn = MSELoss()
        # self.model = PVForecastModel(input_size=nr_features, hidden_size=hidden_size, nr_layers=num_layers, output_size=1)

    def train(self):
        # test if cuda is available, if device is cuda
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available on this machine.")

        min_val_loss = float('inf')
        convergence_iterations = 0
        best_weights = copy.deepcopy(self.model.state_dict())

        train_losses = []
        val_losses = []

        for epoch in range(self.max_epochs):
            # training
            self.model.train()
            epoch_loss = 0.0
            # count = 0
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                # count += 1
                epoch_loss += loss.item()
                # optimizer zero grad clears old gradients from the last step, called before backward
                self.optimizer.zero_grad()
                # loss backward computes the derivative of the loss w.r.t. the parameters
                loss.backward()
                # called after backward() but before optimizer.step() to control gradient magnitude.
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # optimizer step causes the optimizer to take a step based on the gradients of the parameters.
                self.optimizer.step()

            train_losses.append(epoch_loss / len(self.train_loader))

            # validation
            self.model.eval()
            loss = 0.0
            # count = 0
            with torch.no_grad():
                for inputs, targets in self.validation_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss += self.loss_fn(outputs, targets)
                    # count += 1
            loss /= len(self.validation_loader)

            self.lr_scheduler.step(loss)

            val_losses.append(loss.item())

            str_to_print = f"Epoch [{epoch}/{self.max_epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}"

            if loss < min_val_loss:
                min_val_loss = loss
                best_weights = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), '../output/best_model.pth')

                str_to_print += "\t*"
                convergence_iterations = 0
            else:
                convergence_iterations += 1
                if convergence_iterations >= self.early_stopping:
                    print(f"Early stopping after {self.early_stopping} epochs without improvement")
                    break
            print(str_to_print)

        self.model.load_state_dict(best_weights)
        return self.model, train_losses, val_losses

    @torch.no_grad()
    def test(self):
        self.model.eval()

        errors = []
        mean_predicted = []
        mean_actual = []

        for epoch in range(self.max_epochs):
            prediction_error = 0.0
            predicted = 0.0
            actual = 0.0

            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                # store mean actual and predicted values over batch size
                predicted += outputs.cpu().numpy().mean().item()
                actual += targets.cpu().numpy().mean()

                # count += 1
                prediction_error += loss.item()
                # optimizer zero grad clears old gradients from the last step, called before backward
                self.optimizer.zero_grad()
                # loss backward computes the derivative of the loss w.r.t. the parameters
                loss.backward()
                # called after backward() but before optimizer.step() to control gradient magnitude.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # optimizer step causes the optimizer to take a step based on the gradients of the parameters.
                self.optimizer.step()

            errors.append(prediction_error / len(self.test_loader))
            mean_predicted.append(predicted / len(self.test_loader))
            mean_actual.append(actual / len(self.test_loader))

        return errors, mean_predicted, mean_actual