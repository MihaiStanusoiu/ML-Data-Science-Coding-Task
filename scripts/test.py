'''Testing script for PV generation forecasting model.
Script accepts various command-line arguments for configuration, including hyperparams.
Script runs test loop, loading the best model from disk and evaluating on the test dataset by days.
Plots predictions vs actual values, computes eval metrics and prints them.
'''

import argparse
import os
import pickle

import numpy as np

from scripts.datasets import DatasetBuilder
from scripts.pv_generation import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--seq_len', type=int, default=12, help='Sequence length for RNN input')
    parser.add_argument('--hidden_size', type=int, default=32, help='Number of hidden RNN units')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of epochs to train for')
    parser.add_argument('--early_stopping', type=int, default=20, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--device', choices=["cpu", "cuda"], help='Device to use for training', default="cpu")
    args = parser.parse_args()

    dataset_builder = DatasetBuilder(
        pv_history_fp="../data/pv_generation.csv",
        weather_measurements_fp="../data/weather_measurements.csv",
        ghi_forecast_fp="../data/data_ghi_forecast.parquet",
        seq_len=args.seq_len
    )
    dataset = dataset_builder.build_dataset()

    # load scalers
    if os.path.exists("../output/scalerX.pkl"):
        with open("../output/scalerX.pkl", "rb") as f:
            dataset.scalerX = pickle.load(f)

    if os.path.exists("../output/scalerY.pkl"):
        with open("../output/scalerY.pkl", "rb") as f:
            dataset.scalerY = pickle.load(f)

    train_idx = val_idx = test_idx = None
    if os.path.exists("../data/train_idx.npy") and os.path.exists("../data/val_idx.npy") and os.path.exists(
            "../data/test_idx.npy"):
        train_idx = np.load("../data/train_idx.npy")
        val_idx = np.load("../data/val_idx.npy")
        test_idx = np.load("../data/test_idx.npy")
    else:
        train_idx, val_idx, test_idx = dataset_builder.split_dataset(0.8, 0.1)
        np.save("../data/train_idx.npy", train_idx)
        np.save("../data/val_idx.npy", val_idx)
        np.save("../data/test_idx.npy", test_idx)

    train_dataset, val_dataset, test_dataset = dataset_builder.build_split_datasets(train_idx, val_idx, test_idx)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    trainer = Trainer(model_path="../output/best_model.pth", dataset=dataset, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, batch_size=args.batch_size, seq_len=args.seq_len, device=args.device, hidden_size=args.hidden_size, num_features=dataset.number_of_features, num_layers=args.num_layers, max_epochs=args.max_epochs, early_stopping=args.early_stopping)
    metrics = trainer.test()
    print(f"Test R2: {metrics["R2_score"]:.4f}, MAE: {metrics["MAE"]:.4f}, RMSE: {metrics["RMSE"]:.4f}")