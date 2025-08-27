import copy
import pickle
from math import sqrt

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError

from models.pv_forecast_model import PVForecastModel
from scripts.data_analysis import plot_predicted_actual
from scripts.datasets import PVDataset


class Trainer:
    '''
    Class for training and evaluating a PVForecastModel on a given PVDataset
    and train/val/test subsets.
    '''
    def __init__(self, dataset: PVDataset, train_dataset: Subset, val_dataset: Subset, test_dataset: Subset, batch_size: int, seq_len: int, hidden_size: int, num_features: int, num_layers: int, max_epochs: int, early_stopping: int, device: str, model_path: str = None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.device = device

        # Initialize model
        model_factory = lambda : PVForecastModel(input_size=num_features, hidden_size=hidden_size, nr_layers=num_layers,
                                             output_size=1)
        self.model = model_factory()
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path))
            except FileNotFoundError:
                pass

        self.model.to(self.device)

        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.loss_fn = MSELoss()
        # self.model = PVForecastModel(input_size=nr_features, hidden_size=hidden_size, nr_layers=num_layers, output_size=1)

    def train(self):
        '''
        Training algorithm using ADAM and adaptive lr scheduling given the validation loss.
        Stores the best model weights based on validation loss. Includes early stopping
        :return:
        '''
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
            # Training Phase
            self.model.train()
            epoch_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs, _ = self.model(inputs)
                # use only last timestep
                outputs = outputs[:, -1, :]
                target = targets[:, -1, :]
                loss = self.loss_fn(outputs, target)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            train_losses.append(epoch_loss / len(self.train_loader))

            # Validation Phase
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
                    targets = targets[:, -1, :]
                    loss += self.loss_fn(outputs, targets)
            loss /= len(self.validation_loader)

            self.lr_scheduler.step(loss)

            val_losses.append(loss.item())

            str_to_print = f"Epoch [{epoch}/{self.max_epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}"

            if loss < min_val_loss:
                # New best model, store weights
                min_val_loss = loss
                best_weights = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), '../output/best_model.pth')
                with open('../output/scalerX.pkl', 'wb') as f:
                    pickle.dump(self.dataset.scalerX, f)
                with open('../output/scalerY.pkl', 'wb') as f:
                    pickle.dump(self.dataset.scalerY, f)

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
    def test(self) -> dict:
        '''
        Test loop. Iterates over the test set by 24 hour seqeuences, plots predicted vs actual.
        Returns evaluation metrics (R2, MAE, RMSE)
        :return: dict with R2, MAE, RMSE
        '''
        self.model.eval()

        all_preds = []
        all_actuals = []

        for inputs, targets in self.test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs, _ = self.model(inputs)

            predicted_unscaled = self.dataset.scalerY.inverse_transform(outputs.cpu().numpy().squeeze(0))
            actual_unscaled = self.dataset.scalerY.inverse_transform(targets.cpu().numpy().squeeze(0))

            all_preds.append(torch.tensor(predicted_unscaled))
            all_actuals.append(torch.tensor(actual_unscaled))

        # Concatenate all batch predictions and targets
        all_preds_tensor = torch.cat(all_preds).to(self.device)
        all_targets_tensor = torch.cat(all_actuals).to(self.device)

        plot_predicted_actual(all_preds_tensor.cpu().numpy().tolist(), all_targets_tensor.cpu().numpy().tolist())

        r2 = R2Score()
        mae = MeanAbsoluteError()
        mse = MeanSquaredError()

        r2_val = r2(all_preds_tensor, all_targets_tensor).item()
        mae_val = mae(all_preds_tensor, all_targets_tensor).item()
        rmse_val = sqrt(mse(all_preds_tensor, all_targets_tensor).item())
        return {
            "R2_score": r2_val,
            "MAE": mae_val,
            "RMSE": rmse_val
        }