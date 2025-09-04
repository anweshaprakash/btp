"""
lstm_model.py

PyTorch LSTM/GRU models for next-day return prediction.
Two modes:
- baseline LSTM: inputs = past returns (and optional features)
- hybrid LSTM+HMM: inputs = past returns + HMM regime probabilities

Contains:
- TimeSeriesDataset: sliding-window dataset
- LSTMModel: simple stacked LSTM for regression
- train/evaluate functions
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
import os


class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for time series supervised learning.

    X shape -> (seq_len, n_features)
    y -> scalar (next-day return)
    """

    def __init__(self, df: pd.DataFrame, input_cols: List[str], target_col: str, seq_len: int = 20):
        self.seq_len = seq_len
        self.input_cols = input_cols
        self.target_col = target_col
        self.X = df[input_cols].values
        self.y = df[target_col].values
        self.n = len(df)

        # we will create sequences where input is [t - seq_len + 1, ..., t] and target is t+1
        self.indices = []
        for end in range(self.seq_len - 1, self.n - 1):
            self.indices.append(end)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        start = end - self.seq_len + 1
        x = self.X[start:end + 1]  # shape seq_len x n_features
        y = self.y[end + 1]       # next day return
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: batch x seq_len x input_dim
        out, _ = self.lstm(x)  # out: batch x seq_len x hidden
        last = out[:, -1, :]   # take last time step
        return self.fc(last).squeeze(-1)  # shape (batch,)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                n_epochs: int = 50, lr: float = 1e-3, device: str = "cpu", save_path: Optional[str] = None):
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float('inf')
    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        mean_train = np.mean(losses)
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    val_losses.append(((pred - yb) ** 2).mean().item())
            mean_val = np.mean(val_losses)
            print(f"Epoch {epoch}: train_mse={mean_train:.6f}, val_mse={mean_val:.6f}")
            if mean_val < best_val and save_path is not None:
                best_val = mean_val
                torch.save(model.state_dict(), save_path)
        else:
            print(f"Epoch {epoch}: train_mse={mean_train:.6f}")
    return model


def evaluate_model(model: nn.Module, loader: DataLoader, device: str = "cpu") -> dict:
    device = torch.device(device)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.extend(out.tolist())
            trues.extend(yb.numpy().tolist())
    preds = np.array(preds)
    trues = np.array(trues)
    rmse = math.sqrt(mean_squared_error(trues, preds))
    return {"rmse": rmse, "preds": preds, "trues": trues}


if __name__ == "__main__":
    # small local test (needs prepared dataframe)
    print("lstm_model module - define model and training functions")
