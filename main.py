# ============================
# File: main.py
# ============================

import os
from datetime import datetime
import pandas as pd
import numpy as np
import math
import data_pipeline as dp
import news_pipeline as npip
import feature_engineering as fe
import hmm_model as hm
import lstm_model as lm
import visualization as viz
from macroeconomic_pipeline import fetch_macro_data
from timeseries_lstm import RegimeConditionedLSTM, TimeSeriesLSTMDataset, train_regime_lstm

import torch
from torch.utils.data import DataLoader
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------------------------
# 0) Config for Hyperparameter Tuning
# ---------------------------
config = {
    "TICKER": "RELIANCE.NS",
    "START": "2022-01-01",
    "END": "2025-09-18",
    "VOL_WINDOW": 20,
    "N_STATES": 3,
    "SEQ_LEN": 20,
    "BATCH": 32,
    "EPOCHS": 20,
    "LEARNING_RATE": 1e-3,
    "HIDDEN_DIM": 64,
    "NUM_LAYERS": 2,
    "DEVICE": "cpu"
}

OUTDIR = "outputs"
FIGDIR = os.path.join(OUTDIR, "figures")
MODELDIR = os.path.join(OUTDIR, "models")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

# ---------------------------
# 1) Price data & Technical Indicators
# ---------------------------
print("Fetching price data...")
df_price = dp.fetch_price(config["TICKER"], config["START"], config["END"])
df_feat = dp.compute_features(df_price, vol_window=config["VOL_WINDOW"])

# ---------------------------
# 2) News (categorized + sentiment)
# ---------------------------
print("Fetching & processing news...")
query = config["TICKER"].split('.')[0]
df_news = npip.fetch_news_combined(query, max_articles=1000)

if df_news is None or df_news.empty:
    print("⚠️ No news found; continuing with price-only features.")
    df_comb = df_feat.copy()
else:
    df_news['sentiment'] = pd.to_numeric(df_news['sentiment'], errors='coerce').fillna(0.0)
    print(f"News fetched: {len(df_news)} articles. Categories: {df_news['category'].value_counts().to_dict()}")
    df_comb = fe.align_price_news(df_feat, df_news)

# ---------------------------
# 3) Macroeconomic Data
# ---------------------------
print("Fetching macroeconomic data...")
df_macro = fetch_macro_data(config["START"], config["END"])
if not df_macro.empty:
    df_comb = fe.align_macro_data(df_comb, df_macro)
    
# ---------------------------
# 4) Hidden Markov Model (HMM) - Time Series Regime Detection
# ---------------------------
hmm_features = ['ret', 'vol_roll']
print("Fitting rolling HMM on features:", hmm_features)
df_dec = hm.fit_hmm_rolling(df_comb, hmm_features, n_states=config["N_STATES"], window_size=252, min_periods=100)

dates = df_dec.index.tolist()
prices = df_dec['Close'].values
regimes = df_dec['regime'].values

viz.plot_price_regimes(dates, prices, regimes, out_path=os.path.join(FIGDIR, "price_regimes.png"))
regime_probs = df_dec[[f'prob_state_{i}' for i in range(config["N_STATES"])]].values
viz.plot_regime_probabilities(dates, regime_probs, out_path=os.path.join(FIGDIR, "regime_probs.png"))

# ---------------------------
# 5) Prepare datasets for LSTM
# ---------------------------
sentiment_and_regime_cols = [col for col in df_dec.columns if 'sentiment' in col.lower() or 'prob_state' in col.lower() or 'regime' in col.lower()]
for col in sentiment_and_regime_cols:
    df_dec[col] = df_dec[col].shift(1)

baseline_inputs = ['ret', 'ret_ma', 'vol_roll', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bollinger_upper', 'bollinger_lower', 'stoch_k', 'stoch_d']
macro_inputs = list(df_macro.columns) if not df_macro.empty else []

rich_sentiment_features = [
    'sentiment_mean', 'sentiment_pos_count_3d', 'sentiment_neg_count_3d',
    'sentiment_neutral_count_3d', 'sentiment_vol_3d', 'sentiment_vol_7d', 
    'sentiment_momentum_3d', 'news_intensity_3d', 'news_intensity_7d',
    'sentiment_vol_interaction'
]
for cat in ['finance', 'general', 'international', 'national', 'policy', 'geopolitics']:
    rich_sentiment_features.extend([
        f'sentiment_{cat}', f'sentiment_{cat}_strength', f'news_{cat}_ratio', f'sentiment_{cat}_vol_3d', f'news_count_{cat}'
    ])
    rich_sentiment_features.extend([f'finbert_prob_{p}' for p in ['negative', 'neutral', 'positive']])

hybrid_inputs_raw = list(set(baseline_inputs + rich_sentiment_features + macro_inputs))
hybrid_inputs = [col for col in hybrid_inputs_raw if col in df_dec.columns]
missing_cols = [col for col in hybrid_inputs_raw if col not in df_dec.columns]

if missing_cols:
    print(f"⚠️ Skipping {len(missing_cols)} hybrid features because they are not in the DataFrame: {missing_cols}")

df_ready = df_dec.ffill().bfill()
scaler = StandardScaler()

train_end_idx = int(len(df_ready) * 0.7)
df_ready.loc[:, baseline_inputs] = scaler.fit_transform(df_ready[baseline_inputs].fillna(0))
df_ready.loc[:, hybrid_inputs] = scaler.fit_transform(df_ready[hybrid_inputs].fillna(0))

n = len(df_ready)
i_train = int(n * 0.7)
i_val = int(n * 0.85)
df_train = df_ready.iloc[:i_train].copy()
df_val = df_ready.iloc[i_train:i_val].copy()
df_test = df_ready.iloc[i_val:].copy()

print(f"Data split: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test")

# Create data loaders
train_ds_base = lm.TimeSeriesDataset(df_train, baseline_inputs, 'ret', seq_len=config["SEQ_LEN"])
val_ds_base = lm.TimeSeriesDataset(df_val, baseline_inputs, 'ret', seq_len=config["SEQ_LEN"])
test_ds_base = lm.TimeSeriesDataset(df_test, baseline_inputs, 'ret', seq_len=config["SEQ_LEN"])
train_loader_base = DataLoader(train_ds_base, batch_size=config["BATCH"], shuffle=True)
val_loader_base = DataLoader(val_ds_base, batch_size=config["BATCH"], shuffle=False)
test_loader_base = DataLoader(test_ds_base, batch_size=config["BATCH"], shuffle=False)

train_ds_hyb = lm.TimeSeriesDataset(df_train, hybrid_inputs, 'ret', seq_len=config["SEQ_LEN"])
val_ds_hyb = lm.TimeSeriesDataset(df_val, hybrid_inputs, 'ret', seq_len=config["SEQ_LEN"])
test_ds_hyb = lm.TimeSeriesDataset(df_test, hybrid_inputs, 'ret', seq_len=config["SEQ_LEN"])
train_loader_hyb = DataLoader(train_ds_hyb, batch_size=config["BATCH"], shuffle=True)
val_loader_hyb = DataLoader(val_ds_hyb, batch_size=config["BATCH"], shuffle=False)
test_loader_hyb = DataLoader(test_ds_hyb, batch_size=config["BATCH"], shuffle=False)

regime_features = [col for col in [f'prob_state_{i}' for i in range(config["N_STATES"])] if col in df_train.columns]
train_ds_ts = TimeSeriesLSTMDataset(df_train, hybrid_inputs, regime_features, 'ret', config["SEQ_LEN"])
val_ds_ts = TimeSeriesLSTMDataset(df_val, hybrid_inputs, regime_features, 'ret', config["SEQ_LEN"])
test_ds_ts = TimeSeriesLSTMDataset(df_test, hybrid_inputs, regime_features, 'ret', config["SEQ_LEN"])
train_loader_ts = DataLoader(train_ds_ts, batch_size=config["BATCH"], shuffle=True)
val_loader_ts = DataLoader(val_ds_ts, batch_size=config["BATCH"], shuffle=False)
test_loader_ts = DataLoader(test_ds_ts, batch_size=config["BATCH"], shuffle=False)

# ---------------------------
# 6) Training and Evaluation
# ---------------------------

# Train models
print("Training baseline LSTM...")
model_base = lm.LSTMRegressor(input_dim=len(baseline_inputs), hidden_dim=config["HIDDEN_DIM"], num_layers=config["NUM_LAYERS"])
model_base = lm.train_model(model_base, train_loader_base, val_loader_base, n_epochs=config["EPOCHS"], lr=config["LEARNING_RATE"], device=config["DEVICE"], save_path=os.path.join(MODELDIR, "lstm_base.pth"))

print("Training hybrid LSTM...")
model_hyb = lm.LSTMRegressor(input_dim=len(hybrid_inputs), hidden_dim=config["HIDDEN_DIM"], num_layers=config["NUM_LAYERS"])
model_hyb = lm.train_model(model_hyb, train_loader_hyb, val_loader_hyb, n_epochs=config["EPOCHS"], lr=config["LEARNING_RATE"], device=config["DEVICE"], save_path=os.path.join(MODELDIR, "lstm_hyb.pth"))

print("Training Regime-Conditioned LSTM...")
model_ts = RegimeConditionedLSTM(input_dim=len(hybrid_inputs), regime_dim=len(regime_features), hidden_dim=config["HIDDEN_DIM"], num_layers=config["NUM_LAYERS"], output_dim=1)
model_ts = train_regime_lstm(model_ts, train_loader_ts, val_loader_ts, n_epochs=config["EPOCHS"], lr=config["LEARNING_RATE"], device=config["DEVICE"], save_path=os.path.join(MODELDIR, "timeseries_lstm_regime.pth"))

# Evaluate
res_base = lm.evaluate_model(model_base, test_loader_base, device=config["DEVICE"])
res_hyb = lm.evaluate_model(model_hyb, test_loader_hyb, device=config["DEVICE"])
res_ts = lm.evaluate_model(model_ts, test_loader_ts, device=config["DEVICE"])

print("Baseline RMSE:", res_base['rmse'])
print("Hybrid RMSE:", res_hyb['rmse'])
print("Regime-Conditioned RMSE:", res_ts['rmse'])

dates_test = fe.dataset_target_dates(df_test, config["SEQ_LEN"])
viz.plot_predictions(dates_test, res_base['trues'], res_base['preds'], out_path=os.path.join(FIGDIR, "pred_base.png"))
viz.plot_predictions(dates_test, res_hyb['trues'], res_hyb['preds'], out_path=os.path.join(FIGDIR, "pred_hyb.png"))

# ---------------------------
# 7) Save summary
# ---------------------------
with open(os.path.join(OUTDIR, "report.txt"), "w") as f:
    f.write(f"Ticker: {config['TICKER']}\n")
    f.write(f"Baseline RMSE: {res_base['rmse']}\n")
    f.write(f"Hybrid RMSE: {res_hyb['rmse']}\n")
    f.write(f"Regime-Conditioned RMSE: {res_ts['rmse']}\n")
    # OLS is now part of a loop, so let's simplify the output
    f.write("\nFinal OLS Regression: sentiment_mean vs next_ret\n")
    reg_df = df_ready[['sentiment_mean', 'ret']].dropna()
    reg_df['next_ret'] = reg_df['ret'].shift(-1)
    if not reg_df.empty:
        X, y = sm.add_constant(reg_df['sentiment_mean']), reg_df['next_ret']
        ols_model = sm.OLS(y, X).fit()
        f.write(ols_model.summary().as_text())
    else:
        f.write("Not enough data for overall sentiment regression.\n")

print("✅ Done. Outputs in", OUTDIR)
