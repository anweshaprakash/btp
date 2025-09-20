# ============================
# File: lstm_model.py
# ============================
"""
lstm_model.py
- PyTorch LSTM regression model to predict next-day return
- dataset wrapper for sliding windows
- train / evaluate functions
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_cols: List[str], target_col: str, seq_len: int = 20):
        self.seq_len = seq_len
        self.input_cols = input_cols
        self.target_col = target_col
        self.df = df.reset_index(drop=True)
        self.X = self.df[input_cols].values.astype(np.float32)
        self.y = self.df[target_col].values.astype(np.float32)
        self.indices = []
        n = len(self.df)
        for end in range(self.seq_len - 1, n - 1):
            self.indices.append(end)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        start = end - self.seq_len + 1
        x = self.X[start:end+1]  # seq_len x features
        y = self.y[end+1]        # next day
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        # x: batch x seq_len x input_dim
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                n_epochs: int = 30, lr: float = 1e-3, device: str = "cpu", save_path: Optional[str] = None):
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
        if val_loader:
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
            if mean_val < best_val and save_path:
                best_val = mean_val
                torch.save(model.state_dict(), save_path)
        else:
            print(f"Epoch {epoch}: train_mse={mean_train:.6f}")
    return model

def evaluate_model(model: nn.Module, loader: DataLoader, device: str = "cpu"):
    """
    Evaluates a given model using a data loader.
    This function has been made more robust to handle different DataLoader
    outputs, including the three-value output from TimeSeriesLSTMDataset.
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for item in loader:
            if len(item) == 2:
                # Normal LSTM model
                xb, yb = item
                out = model(xb).cpu().numpy()
                trues_batch = yb.numpy().tolist()
            elif len(item) == 3:
                # Regime-conditioned LSTM model
                feature_seq, regime_context, target = item
                out = model(feature_seq, regime_context).cpu().numpy()
                trues_batch = target.numpy().tolist()
            else:
                raise ValueError("Unexpected number of values in DataLoader output.")

            preds.extend(out.tolist())
            trues.extend(trues_batch)

    preds = np.array(preds)
    trues = np.array(trues)
    rmse = math.sqrt(mean_squared_error(trues, preds))
    return {"rmse": rmse, "preds": preds, "trues": trues}

if __name__ == "__main__":
    print("LSTM model module. Import and use train_model/evaluate_model.")
