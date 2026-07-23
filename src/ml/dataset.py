"""
QuantLab AI Dataset Pipeline
Downloads stock data, scales it, creates sliding windows,
and prepares PyTorch DataLoaders.
"""

import random
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from .config import config


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(config.random_seed)


class StockDataset(Dataset):

    def __init__(self, sequences, labels):

        self.x = torch.tensor(
            sequences,
            dtype=torch.float32
        )

        self.y = torch.tensor(
    labels,
    dtype=torch.float32
).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DataProcessor:

    def __init__(self):

        self.scaler = MinMaxScaler()

    def download_data(self):

        df = yf.download(
            config.ticker,
            start=config.start_date,
            end=config.end_date,
            progress=False
        )

        if len(df) == 0:
            raise ValueError("No stock data downloaded.")

        df = df[["Close"]].dropna()

        return df

    def preprocess(self, df):

        scaled = self.scaler.fit_transform(df)

        return scaled

    def create_sequences(self, data):

        x = []
        y = []

        window = config.window_size

        for i in range(window, len(data)):
            x.append(data[i-window:i])
            y.append(data[i][0])

        return np.array(x), np.array(y)

    def split(self, x, y):

        split_idx = int(len(x) * config.train_split)

        x_train = x[:split_idx]
        y_train = y[:split_idx]

        x_test = x[split_idx:]
        y_test = y[split_idx:]

        return (
            x_train,
            y_train,
            x_test,
            y_test
        )

    def loaders(self):

        df = self.download_data()

        scaled = self.preprocess(df)

        x, y = self.create_sequences(scaled)

        (
            x_train,
            y_train,
            x_test,
            y_test
        ) = self.split(x, y)

        train_dataset = StockDataset(
            x_train,
            y_train
        )

        test_dataset = StockDataset(
            x_test,
            y_test
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        return (
            train_loader,
            test_loader,
            self.scaler,
            df
        )


def get_dataloaders():

    processor = DataProcessor()

    return processor.loaders()
