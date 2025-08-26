from typing import List

import pandas as pd
import torch
from mpmath.calculus.calculus import defun
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split, Subset
import numpy as np

from scripts.data_analysis import compute_statistics, plot_timestamped_data, analyze_pv_history, \
    analyze_weather_measurements


class PVDataset(Dataset):
    '''
    PV generation forecasting dataset class. Samples are feature sequences of length seq_len.
    Targets are PV generation values 2 hours ahead.
    '''
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int, scalerX: StandardScaler = None,
                 scalerY: StandardScaler = None, shuffle: bool = False, exclude_leys: List | None = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.number_of_features = self.features.shape[1]
        self.indices = np.arange(len(self.targets))
        self.exclude_keys = exclude_leys
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.targets - self.seq_len)

    def __getitem__(self, idx):
        # # select sequence of len self.seq_len for RNN
        # if idx < self.seq_len - 1:
        #     # pad with zeros
        #     pad_size = self.seq_len - 1 - idx
        #     feature_seq = torch.cat((torch.zeros((pad_size, self.number_of_features)), self.features[:idx + 1]), dim=0)
        # else:
        #     feature_seq = torch.zeros((self.seq_len, self.number_of_features))
        #     feature_seq = self.features[idx - self.seq_len + 1:idx + 1]
        # target = self.targets[idx]
        # return feature_seq, target
        end = self.indices[idx]
        start = end - self.seq_len + 1
        if start < 0:
            start = 0
        len = end - start + 1
        feature_seq = torch.zeros((self.seq_len, self.number_of_features))
        feature_seq[self.seq_len - len:] = self.features[start:end + 1]
        target = self.targets[end]

        # feature_seq[:, :5] = torch.tensor(self.scalerX.transform(feature_seq[:, :5]), dtype=torch.float32) if self.scalerX else feature_seq
        # target = torch.tensor(self.scalerY.transform(target.reshape(1, -1)).flatten(), dtype=torch.float32) if self.scalerY else target

        return feature_seq, target

class PVDayDataset(Dataset):
    def __init__(self, timestamps: np.ndarray, features: np.ndarray, targets: np.ndarray):
        self.timestamps = timestamps
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

        # index by days: store unique days
        self.days = self.timestamps.dt.day_of_year.unique()

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        features = self.features[self.timestamps.dt.day_of_year.values == self.days[idx]]
        targets = self.targets[self.timestamps.dt.day_of_year.values == self.days[idx]]

        return features, targets


class DatasetBuilder:
    '''
    Dataset builder class. Includes: preprocessing; normalization; merging
    and synchronization of pv_history, weather forecast; splitting methods.
    '''
    def __init__(self, pv_history_fp, weather_measurements_fp, ghi_forecast_fp, seq_len):
        self.pv_history_fp = pv_history_fp
        self.weather_measurements_fp = weather_measurements_fp
        self.ghi_forecast_fp = ghi_forecast_fp
        self.seq_len = seq_len

    def build_dataset(self):
        '''
        Preprocess subdatasets, synchronize and merge them into one dataset.
        :return:
        dataset: PVDataset object
        '''
        self.pv_history_raw = pd.read_csv(self.pv_history_fp, parse_dates=['time'])
        self.weather_measurements_raw = pd.read_csv(self.weather_measurements_fp, parse_dates=['time'])
        self.ghi_forecasts_raw = pd.read_parquet(self.ghi_forecast_fp)

        self.preprocess_pv_history()
        self.preprocess_weather_measurements()
        self.preprocess_ghi_forecasts()

        datasetDf = self.merge_datasets()

        self.feature_cols_without_time_features = ["pv_generation", "cglo", "ff", "tl", "ghi_forecast"]

        self.feature_cols = ["pv_generation", "cglo", "ff", "tl", "ghi_forecast", "hour_cos_pv", "hour_sin_pv",
                             "day_cos_pv", "day_sin_pv", "hour_cos_weather", "hour_sin_weather", "day_cos_weather",
                             "day_sin_weather", "hour_cos_ghi", "hour_sin_ghi", "day_cos_ghi", "day_sin_ghi"]


        self.number_of_features = len(self.feature_cols)

        self.X = datasetDf[self.feature_cols]
        self.Y = datasetDf[["targets"]]

        # get timestamps column
        self.timestamps = datasetDf["time"]

        self.dataset = PVDataset(self.X.values, self.Y.values,  seq_len=self.seq_len)
        return self.dataset

    def split_dataset(self, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Splits dataset indices into train, validation and test sets, given input ratios.
        :param train_ratio: float
        :param val_ratio: float
        :return:
        train_idx, val_idx, test_idx: np.ndarray
        '''
        indices = np.arange(len(self.Y))
        test_ratio = 1 - val_ratio - train_ratio

        total_sequences = len(self.Y) - self.seq_len
        train_split = int(train_ratio * total_sequences)
        val_split = int((train_ratio + val_ratio) * total_sequences)
        test_split = int((train_ratio + val_ratio + test_ratio) * total_sequences)

        self.train_idx, self.val_idx, self.test_idx = list(range(train_split)), list(range(train_split, total_sequences)), list(range(val_split, total_sequences))
        return self.train_idx, self.val_idx, self.test_idx

    def build_split_datasets(self, train_idx: list, val_idx: list, test_idx: list) -> tuple[Subset, Subset, Subset]:
        '''
        :param train_idx: int
        :param val_idx: int
        :param test_idx: int
        :return:
        train_dataset, validation_dataset, test_dataset: PVDataset
        '''

        self.train_subset = Subset(self.dataset, train_idx)
        self.validation_subset = Subset(self.dataset, val_idx)
        self.test_subset = Subset(self.dataset, test_idx)

        self.X_train, self.Y_train = self.X.iloc[train_idx], self.Y.iloc[train_idx]
        self.X_validation, self.Y_validation = self.X.iloc[val_idx], self.Y.iloc[val_idx]
        self.X_test, self.Y_test = self.X.iloc[test_idx], self.Y.iloc[test_idx]

        train_timestamps = self.timestamps.iloc[train_idx]
        val_timestamps = self.timestamps.iloc[val_idx]
        test_timestamps = self.timestamps.iloc[test_idx]

        # self.X_train, self.X_temp, self.Y_train, self.Y_temp = train_test_split(self.X, self.Y, train_size=train_split, shuffle=False)
        # val_ratio_adjusted = val_split / (1 - train_split)
        # self.X_validation, self.X_test, self.Y_validation, self.Y_test = train_test_split(self.X_temp, self.Y_temp, train_size=val_ratio_adjusted, shuffle=False)

        # self.X_train = self.train_dataset[self.feature_cols]
        # self.Y_train = self.train_dataset["targets"]
        #
        # self.X_validation = self.validation_dataset[self.feature_cols]
        # self.Y_validation = self.validation_dataset["targets"]
        #
        # self.X_test = self.test_dataset[self.feature_cols]
        # self.Y_test = self.test_dataset["targets"]

        scalerX, scalerY = self.normalize_data()
        self.dataset.scalerX = scalerX
        self.dataset.scalerY = scalerY

        return (Subset(self.dataset, train_idx),
                Subset(self.dataset, val_idx),
                Subset(self.dataset, test_idx))


    def normalize_data(self) -> tuple[StandardScaler, StandardScaler]:
        '''
        Normalize data with mean 0 and std 1 scaler fit on training set.
        :return: scalerX, scalerY: StandardScaler
        '''
        self.scalerX = StandardScaler()
        self.scalerX.fit(self.X_train[self.feature_cols_without_time_features])
        self.X_train[self.feature_cols_without_time_features] = self.scalerX.transform(self.X_train[self.feature_cols_without_time_features])
        self.X_validation[self.feature_cols_without_time_features] = self.scalerX.transform(self.X_validation[self.feature_cols_without_time_features])
        self.X_test[self.feature_cols_without_time_features] = self.scalerX.transform(self.X_test[self.feature_cols_without_time_features])

        self.scalerY = StandardScaler()
        self.scalerY.fit(self.Y_train)
        self.Y_train = self.scalerY.transform(self.Y_train)
        self.Y_validation = self.scalerY.transform(self.Y_validation)
        self.Y_test = self.scalerY.transform(self.Y_test)

        self.X_train = self.X_train.values
        self.X_validation = self.X_validation.values
        self.X_test = self.X_test.values

        self.dataset.features[:, :5] = torch.tensor(self.scalerX.transform(self.dataset.features[:, :5]), dtype=torch.float32)
        self.dataset.targets = torch.tensor(self.scalerY.transform(self.dataset.targets), dtype=torch.float32)

        return self.scalerX, self.scalerY

    def preprocess_pv_history(self):
        '''
        Preprocess pv_history dataset. Filter out negative values and outliers.
        Add time-based features as continuous signals: hour, day of year
        :return:
        '''
        # set timestamp as index
        # set negative values to NA
        self.pv_history = self.pv_history_raw.copy()
        self.pv_history[self.pv_history['pv_generation'] < 0] = np.nan
        Q1 = self.pv_history["pv_generation"].quantile(0.25)
        Q3 = self.pv_history["pv_generation"].quantile(0.75)
        IQR = Q3 - Q1

        # Keep only within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        self.pv_history[(self.pv_history["pv_generation"] < Q1 - 1.5 * IQR) | (self.pv_history["pv_generation"] > Q3 + 1.5 * IQR)] = np.nan

        self.pv_history["hour"] = self.pv_history["time"].dt.hour
        self.pv_history["hour_cos"] = np.cos(2 * np.pi * self.pv_history["time"].dt.hour / 24)
        self.pv_history["hour_sin"] = np.sin(2 * np.pi * self.pv_history["time"].dt.hour / 24)
        self.pv_history["day"] = self.pv_history["time"].dt.day_of_year
        self.pv_history["day_cos"] = np.cos(2 * np.pi * self.pv_history["time"].dt.day_of_year / 365)
        self.pv_history["day_sin"] = np.sin(2 * np.pi * self.pv_history["time"].dt.day_of_year / 365)

    def preprocess_weather_measurements(self):
        '''
        Preprocess weather_measurements dataset.
        Add time-based features as continuous signals: hour, day of year
        :return:
        '''
        # set timestamp as index
        self.weather_measurements = self.weather_measurements_raw.copy()
        self.weather_measurements["hour"] = self.weather_measurements["time"].dt.hour
        self.weather_measurements["hour_cos"] = np.cos(2 * np.pi * self.weather_measurements["time"].dt.hour / 24)
        self.weather_measurements["hour_sin"] = np.sin(2 * np.pi * self.weather_measurements["time"].dt.hour / 24)
        self.weather_measurements["day"] = self.weather_measurements["time"].dt.day_of_year
        self.weather_measurements["day_cos"] = np.cos(
            2 * np.pi * self.weather_measurements["time"].dt.day_of_year / 365)
        self.weather_measurements["day_sin"] = np.sin(
            2 * np.pi * self.weather_measurements["time"].dt.day_of_year / 365)

    def preprocess_ghi_forecasts(self):
        '''
        Preprocess ghi_forecasts dataset.
        Select most recent forecast with at least 2 hours delta time for each datapoint.
        :return:
        '''
        self.ghi_forecasts = self.ghi_forecasts_raw.copy()
        valid_time = self.ghi_forecasts.index.get_level_values(0)
        time_delta = self.ghi_forecasts.index.get_level_values(1)

        delta_2_h = pd.Timedelta(hours=2)
        self.ghi_forecasts = self.ghi_forecasts[time_delta >= delta_2_h]

        grouped_ghi = self.ghi_forecasts.groupby(level=0)

        # for each group, select data with smallest time_delta (index level 1)
        self.ghi_forecasts = grouped_ghi.apply(lambda x: x.xs(x.index.get_level_values(1).min(), level=1))
        # remove timedelta index
        self.ghi_forecasts.index = self.ghi_forecasts.index.droplevel(1)

        # add time-based features
        self.ghi_forecasts["time"] = self.ghi_forecasts.index
        self.ghi_forecasts["hour"] = self.ghi_forecasts["time"].dt.hour
        self.ghi_forecasts["hour_cos"] = np.cos(2 * np.pi * self.ghi_forecasts["time"].dt.hour / 24)
        self.ghi_forecasts["hour_sin"] = np.sin(2 * np.pi * self.ghi_forecasts["time"].dt.hour / 24)
        self.ghi_forecasts["day"] = self.ghi_forecasts["time"].dt.day_of_year
        self.ghi_forecasts["day_cos"] = np.cos(
            2 * np.pi * self.ghi_forecasts["time"].dt.day_of_year / 365)
        self.ghi_forecasts["day_sin"] = np.sin(
            2 * np.pi * self.ghi_forecasts["time"].dt.day_of_year / 365)

    def merge_datasets(self) -> pd.DataFrame:
        '''
        Join datasets on timestamp index.
        :return: pd.DataFrame
        '''
        # merge datasets on timestamp
        self.pv_history.set_index('time', inplace=True)
        self.weather_measurements.set_index('time', inplace=True)

        # set targets
        self.pv_history["targets"] = self.pv_history['pv_generation'].shift(-2)
        self.pv_history = self.pv_history.dropna(subset=["targets"])

        self.pv_history.sort_index()
        self.weather_measurements.sort_index()
        merged_dataset = self.pv_history.join(self.weather_measurements, how="inner", lsuffix="_pv", rsuffix="_weather")
        merged_dataset = merged_dataset.join(self.ghi_forecasts, how="inner", rsuffix="_ghi")

        # drop na
        merged_dataset = merged_dataset.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        # rename hour_cos, hour_sin, day_cos, day_sin columns to hour_cos_ghi, hour_sin_ghi, day_cos_ghi, day_sin_ghi
        merged_dataset.rename(columns={"hour_cos": "hour_cos_ghi", "hour_sin": "hour_sin_ghi", "day_cos": "day_cos_ghi", "day_sin": "day_sin_ghi"}, inplace=True)
        merged_dataset.reset_index(inplace=True)
        return merged_dataset

if __name__ == "__main__":
    dataset_builder = DatasetBuilder(
        pv_history_fp="../data/pv_generation.csv",
        weather_measurements_fp="../data/weather_measurements.csv",
        ghi_forecast_fp="../data/data_ghi_forecast.parquet",
        seq_len=12
    )


    dataset = dataset_builder.build_dataset()

    analyze_pv_history(dataset_builder.pv_history_raw)
    analyze_weather_measurements(dataset_builder.weather_measurements_raw)