"""
main.py

Run the full pipeline:
- Fetch data
- Preprocess features
- Train HMM and decode regimes
- Build datasets for LSTM baseline and hybrid (with HMM probs)
- Train both models and evaluate RMSE
- Save plots to outputs/figures
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

import data_pipeline as dp
import hmm_model as hm
import lstm_model as lm
import visualization as viz

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

# ----------------------------
# User-configurable parameters
# ----------------------------
TICKER = "SPY"               # or "BTC-USD", "AAPL"
START = "2015-01-01"
END = "2025-08-31"
VOL_WINDOW = 20
HMM_FEATURES = ['ret', 'vol_roll']  # features used to train HMM
N_STATES = 3
SEQ_LEN = 20
BATCH_SIZE = 64
EPOCHS = 30
DEVICE = "cpu"  # or "cuda" if available
OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# 1) Fetch + preprocess
# ----------------------------
print("Fetching data...")
df_price = dp.fetch_data(TICKER, START, END)
df = dp.compute_features(df_price, vol_window=VOL_WINDOW, normalize=False)
# drop NaNs if any remain
df = df.dropna().copy()

# ----------------------------
# 2) HMM baseline
# ----------------------------
print("Fitting HMM...")
hmm_model, scaler = hm.fit_hmm(df, HMM_FEATURES, n_states=N_STATES)
df_dec = hm.decode_regimes(hmm_model, scaler, df, HMM_FEATURES)

# Save model
hm.save_model(hmm_model, scaler, os.path.join(MODEL_DIR, f"hmm_{TICKER}.pkl"))

# ----------------------------
# 3) Visualize HMM results
# ----------------------------
viz.plot_price_with_regimes(df_dec, close_col='Close', regime_col='regime', out_path=os.path.join(FIG_DIR, "price_regimes.png"))
viz.plot_regime_probabilities(df_dec, prob_prefix='prob_state_', out_path=os.path.join(FIG_DIR, "regime_probs.png"))
print("Saved regime plots to", FIG_DIR)

# ----------------------------
# 4) Prepare datasets for LSTM
# ----------------------------
# baseline inputs: use 'ret' (and ret_ma or vol) as features
# hybrid inputs: include HMM state probabilities
df_for_model = df_dec.copy()
# fill any small NaNs
df_for_model = df_for_model.fillna(method='ffill').fillna(method='bfill')

# Choose baseline and hybrid input columns
baseline_inputs = ['ret']  # you can add 'ret_ma', 'vol_roll' if desired
hybrid_inputs = baseline_inputs + [f'prob_state_{i}' for i in range(N_STATES)]

target_col = 'ret'  # next-day return (we will predict scalar return)

# split into train/val/test by date (time-series split)
train_ratio = 0.7
val_ratio = 0.15
n = len(df_for_model)
i_train = int(n * train_ratio)
i_val = int(n * (train_ratio + val_ratio))

df_train = df_for_model.iloc[:i_train]
df_val = df_for_model.iloc[i_train:i_val]
df_test = df_for_model.iloc[i_val:]

# create datasets
train_ds_baseline = lm.TimeSeriesDataset(df_train, baseline_inputs, target_col, seq_len=SEQ_LEN)
val_ds_baseline = lm.TimeSeriesDataset(df_val, baseline_inputs, target_col, seq_len=SEQ_LEN)
test_ds_baseline = lm.TimeSeriesDataset(df_test, baseline_inputs, target_col, seq_len=SEQ_LEN)

train_loader_base = DataLoader(train_ds_baseline, batch_size=BATCH_SIZE, shuffle=True)
val_loader_base = DataLoader(val_ds_baseline, batch_size=BATCH_SIZE, shuffle=False)
test_loader_base = DataLoader(test_ds_baseline, batch_size=BATCH_SIZE, shuffle=False)

# hybrid
train_ds_hybrid = lm.TimeSeriesDataset(df_train, hybrid_inputs, target_col, seq_len=SEQ_LEN)
val_ds_hybrid = lm.TimeSeriesDataset(df_val, hybrid_inputs, target_col, seq_len=SEQ_LEN)
test_ds_hybrid = lm.TimeSeriesDataset(df_test, hybrid_inputs, target_col, seq_len=SEQ_LEN)

train_loader_hybrid = DataLoader(train_ds_hybrid, batch_size=BATCH_SIZE, shuffle=True)
val_loader_hybrid = DataLoader(val_ds_hybrid, batch_size=BATCH_SIZE, shuffle=False)
test_loader_hybrid = DataLoader(test_ds_hybrid, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 5) Train baseline LSTM
# ----------------------------
print("Training baseline LSTM...")
input_dim_base = len(baseline_inputs)
model_base = lm.LSTMModel(input_dim=input_dim_base, hidden_dim=64, num_layers=2)
save_path_base = os.path.join(MODEL_DIR, "lstm_base.pth")
model_base = lm.train_model(model_base, train_loader_base, val_loader_base, n_epochs=EPOCHS, lr=1e-3, device=DEVICE, save_path=save_path_base)

# ----------------------------
# 6) Train hybrid LSTM
# ----------------------------
print("Training hybrid LSTM (with HMM probs)...")
input_dim_hybrid = len(hybrid_inputs)
model_hybrid = lm.LSTMModel(input_dim=input_dim_hybrid, hidden_dim=64, num_layers=2)
save_path_hybrid = os.path.join(MODEL_DIR, "lstm_hybrid.pth")
model_hybrid = lm.train_model(model_hybrid, train_loader_hybrid, val_loader_hybrid, n_epochs=EPOCHS, lr=1e-3, device=DEVICE, save_path=save_path_hybrid)

# ----------------------------
# 7) Evaluate on test set
# ----------------------------
print("Evaluating baseline...")
res_base = lm.evaluate_model(model_base, test_loader_base, device=DEVICE)
print("Baseline RMSE:", res_base['rmse'])

print("Evaluating hybrid...")
res_hybrid = lm.evaluate_model(model_hybrid, test_loader_hybrid, device=DEVICE)
print("Hybrid RMSE:", res_hybrid['rmse'])

# Save prediction plots (align dates)
# Note: TimeSeriesDataset maps each sample to the date corresponding to 'end+1' (target day)
def get_sample_dates(df_slice: pd.DataFrame, seq_len: int):
    # returns target dates for dataset constructed as above
    dates = df_slice.index.to_list()
    target_dates = []
    for end in range(seq_len - 1, len(dates) - 1):
        target_dates.append(dates[end + 1])  # target day
    return target_dates

test_dates = get_sample_dates(df_test, SEQ_LEN)

viz.plot_predictions_vs_actual(test_dates, res_base['trues'], res_base['preds'], out_path=os.path.join(FIG_DIR, "pred_base.png"))
viz.plot_predictions_vs_actual(test_dates, res_hybrid['trues'], res_hybrid['preds'], out_path=os.path.join(FIG_DIR, "pred_hybrid.png"))

print("Saved prediction plots to", FIG_DIR)

# ----------------------------
# 8) Evaluate regime classification accuracy (proxy)
# ----------------------------
# There is no ground-truth for regimes. We'll create a simple heuristic "label":
# bull if ret_ma > mean + std, bear if ret_ma < mean - std, else sideways.
print("Evaluating regime accuracy vs heuristic (proxy)...")
ret_ma = df_for_model['ret_ma']
mu = ret_ma.mean()
sigma = ret_ma.std()
def heuristic_label(row):
    if row['ret_ma'] > mu + 0.5 * sigma:
        return 0  # bull
    elif row['ret_ma'] < mu - 0.5 * sigma:
        return 2  # bear
    else:
        return 1  # sideways

df_for_model['heur_label'] = df_for_model.apply(heuristic_label, axis=1)
# Align lengths & compare on test slice
test_df = df_for_model.iloc[i_val:].copy()
# If necessary drop initial rows for sequence length
test_df = test_df.iloc[SEQ_LEN:]  # because dataset target starts after seq_len
# compute accuracy between HMM regime and heuristic
acc = (test_df['regime'].values == test_df['heur_label'].values).mean()
print(f"Regime proxy accuracy (test window): {acc:.4f}")

# Save a quick summary
with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
    f.write(f"Ticker: {TICKER}\n")
    f.write(f"HMM states: {N_STATES}\n")
    f.write(f"Baseline RMSE: {res_base['rmse']:.6f}\n")
    f.write(f"Hybrid RMSE: {res_hybrid['rmse']:.6f}\n")
    f.write(f"Regime proxy accuracy: {acc:.4f}\n")

print("Done. Outputs saved to", OUTPUT_DIR)
