import copy
from math import sqrt

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError

from models.pv_forecast_model import PVForecastModel
from scripts.datasets import PVDataset


class Trainer:
    def __init__(self, dataset: PVDataset, train_dataset: Subset, val_dataset: Subset, test_dataset: Subset, batch_size: int, seq_len: int, hidden_size: int, num_features: int, num_layers: int, max_epochs: int, early_stopping: int, device: str, model_path: str = None):
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

        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
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
                outputs, _ = self.model(inputs)
                # use only last timestep
                outputs = outputs[:, -1, :]
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
                    outputs, _ = self.model(inputs)
                    # use only last timestep
                    outputs = outputs[:, -1, :]
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

        all_preds = []
        all_actuals = []

        for inputs, targets in self.test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs, _ = self.model(inputs)
            # use only last timestep
            outputs = outputs[:, -1, :]

            predicted_unscaled = self.dataset.scalerY.inverse_transform(outputs.cpu().numpy())
            actual_unscaled = self.dataset.scalerY.inverse_transform(targets)

            all_preds.append(torch.tensor(predicted_unscaled))
            all_actuals.append(torch.tensor(actual_unscaled))

        # Concatenate all batch predictions and targets
        all_preds_tensor = torch.cat(all_preds).to(self.device)
        all_targets_tensor = torch.cat(all_actuals).to(self.device)

        r2 = R2Score()
        mae = MeanAbsoluteError()
        mse = MeanSquaredError()

        r2_val = r2(all_preds_tensor, all_targets_tensor).item()
        mae_val = mae(all_preds_tensor, all_targets_tensor).item()
        rmse_val = sqrt(mse(all_preds_tensor, all_targets_tensor).item())
        return r2_val, mae_val, rmse_val