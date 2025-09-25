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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

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

# Additional evaluation metrics
def compute_additional_metrics(trues: np.ndarray, preds: np.ndarray) -> dict:
    trues = np.asarray(trues).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    rmse = math.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    # Avoid division by zero in MAPE
    eps = 1e-9
    mape = float(np.mean(np.abs((trues - preds) / np.clip(np.abs(trues), eps, None))) * 100.0)
    try:
        r2 = r2_score(trues, preds)
    except Exception:
        r2 = float('nan')
    # Directional accuracy (% of correct sign predictions)
    dir_acc = float(np.mean(np.sign(preds) == np.sign(trues)) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "direction_accuracy": dir_acc}

def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2):
    """Simple Diebold-Mariano test for equal predictive accuracy.
    e1, e2: forecast errors (y - yhat). For squared error loss (power=2).
    Returns (DM_stat, p_value).
    Note: Uses variance of loss differential without HAC for h=1.
    """
    e1 = np.asarray(e1).reshape(-1)
    e2 = np.asarray(e2).reshape(-1)
    # Align lengths and drop NaNs
    n_raw = min(len(e1), len(e2))
    if n_raw < 5:
        return float('nan'), float('nan')
    e1 = e1[:n_raw]
    e2 = e2[:n_raw]
    mask = (~np.isnan(e1)) & (~np.isnan(e2))
    e1 = e1[mask]
    e2 = e2[mask]
    if len(e1) < 5:
        return float('nan'), float('nan')
    if power == 1:
        d = np.abs(e1) - np.abs(e2)
    else:
        d = (e1 ** power) - (e2 ** power)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 5:
        return float('nan'), float('nan')
    d_bar = np.mean(d)
    # For h=1, use sample variance; for h>1, a full HAC should be used
    var_d = np.var(d, ddof=1)
    dm_stat = d_bar / math.sqrt(var_d / n + 1e-12)
    # Two-sided p-value using t distribution with n-1 df
    p_val = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_val)

# Compute metrics for all three models
metrics_base = compute_additional_metrics(res_base['trues'], res_base['preds'])
metrics_hyb = compute_additional_metrics(res_hyb['trues'], res_hyb['preds'])
metrics_ts = compute_additional_metrics(res_ts['trues'], res_ts['preds'])

# Per-regime metrics on the test set
# Align regimes with test target dates
regimes_test_series = df_test['regime'].iloc[config["SEQ_LEN"]:].values if 'regime' in df_test.columns else None
per_regime_metrics = {}
if regimes_test_series is not None:
    # Ensure equal length alignment across all arrays
    len_base = len(res_base['trues'])
    len_hyb = len(res_hyb['trues'])
    len_ts = len(res_ts['trues'])
    min_len = min(len(regimes_test_series), len_base, len_hyb, len_ts)
    if min_len > 0:
        reg_aligned = np.asarray(regimes_test_series)[:min_len]
        base_t = np.asarray(res_base['trues'])[:min_len]
        base_p = np.asarray(res_base['preds'])[:min_len]
        hyb_t = np.asarray(res_hyb['trues'])[:min_len]
        hyb_p = np.asarray(res_hyb['preds'])[:min_len]
        ts_t = np.asarray(res_ts['trues'])[:min_len]
        ts_p = np.asarray(res_ts['preds'])[:min_len]
        for k in range(config["N_STATES"]):
            mask = (reg_aligned == k)
            if np.any(mask):
                per_regime_metrics[k] = {
                    'baseline_rmse': float(math.sqrt(mean_squared_error(base_t[mask], base_p[mask]))),
                    'hybrid_rmse': float(math.sqrt(mean_squared_error(hyb_t[mask], hyb_p[mask]))),
                    'regime_lstm_rmse': float(math.sqrt(mean_squared_error(ts_t[mask], ts_p[mask])))
                }

# DM tests (squared error loss) between models
e_base = res_base['trues'] - res_base['preds']
e_hyb = res_hyb['trues'] - res_hyb['preds']
e_ts = res_ts['trues'] - res_ts['preds']
dm_hyb_vs_base = diebold_mariano_test(e_hyb, e_base, h=1, power=2)
dm_ts_vs_base = diebold_mariano_test(e_ts, e_base, h=1, power=2)
dm_ts_vs_hyb = diebold_mariano_test(e_ts, e_hyb, h=1, power=2)

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
    f.write("\n== Overall Metrics ==\n")
    f.write(f"Baseline: RMSE={metrics_base['rmse']:.6f}, MAE={metrics_base['mae']:.6f}, MAPE={metrics_base['mape']:.2f}%, R2={metrics_base['r2']:.4f}, DirectionAcc={metrics_base['direction_accuracy']:.2f}%\n")
    f.write(f"Hybrid:   RMSE={metrics_hyb['rmse']:.6f}, MAE={metrics_hyb['mae']:.6f}, MAPE={metrics_hyb['mape']:.2f}%, R2={metrics_hyb['r2']:.4f}, DirectionAcc={metrics_hyb['direction_accuracy']:.2f}%\n")
    f.write(f"Regime LSTM: RMSE={metrics_ts['rmse']:.6f}, MAE={metrics_ts['mae']:.6f}, MAPE={metrics_ts['mape']:.2f}%, R2={metrics_ts['r2']:.4f}, DirectionAcc={metrics_ts['direction_accuracy']:.2f}%\n")

    # DM tests
    f.write("\n== Diebold-Mariano Tests (squared error loss) ==\n")
    f.write(f"Hybrid vs Baseline: DM={dm_hyb_vs_base[0]:.4f}, p-value={dm_hyb_vs_base[1]:.6f}\n")
    f.write(f"Regime LSTM vs Baseline: DM={dm_ts_vs_base[0]:.4f}, p-value={dm_ts_vs_base[1]:.6f}\n")
    f.write(f"Regime LSTM vs Hybrid: DM={dm_ts_vs_hyb[0]:.4f}, p-value={dm_ts_vs_hyb[1]:.6f}\n")

    # Per-regime
    if per_regime_metrics:
        f.write("\n== Per-Regime RMSE (test set) ==\n")
        for k, vals in per_regime_metrics.items():
            f.write(f"Regime {k}: Baseline RMSE={vals['baseline_rmse']:.6f}, Hybrid RMSE={vals['hybrid_rmse']:.6f}, Regime-LSTM RMSE={vals['regime_lstm_rmse']:.6f}\n")
    # OLS benchmark: sentiment_mean -> next day's return (next_ret)
    f.write("\nFinal OLS Regression: sentiment_mean -> next_ret (benchmark)\n")
    if 'sentiment_mean' in df_ready.columns:
        reg_df = df_ready[['sentiment_mean', 'ret']].copy()
        # Target is next day's return
        reg_df['next_ret'] = reg_df['ret'].shift(-1)
        # Drop rows with NaNs in predictors or target
        reg_df = reg_df.dropna(subset=['sentiment_mean', 'next_ret'])
        # Ensure numeric and finite
        reg_df = reg_df[np.isfinite(reg_df['sentiment_mean']) & np.isfinite(reg_df['next_ret'])]
        # Run OLS only if sufficient variation and samples
        if len(reg_df) >= 30 and reg_df['sentiment_mean'].var() > 0 and reg_df['next_ret'].var() > 0:
            X = sm.add_constant(reg_df['sentiment_mean'].astype(float))
            y = reg_df['next_ret'].astype(float)
            try:
                ols_model = sm.OLS(y, X).fit()
                f.write(ols_model.summary().as_text())
            except Exception as e:
                f.write(f"OLS failed: {e}\n")
        else:
            f.write("Skipping OLS: insufficient samples or variance (need >=30 samples and non-zero variance).\n")
    else:
        f.write("Skipping OLS: 'sentiment_mean' not found in dataframe.\n")

    # ---------------------------
    # 8) Descriptive Tables for LaTeX (real values)
    # ---------------------------
    f.write("\n== Technical Indicators Summary ==\n")
    tech_cols = [
        ('rsi', 'RSI'),
        ('macd', 'MACD'),
        ('bollinger_upper', 'Bollinger Upper'),
        ('bollinger_lower', 'Bollinger Lower'),
        ('stoch_k', 'Stochastic %K'),
        ('stoch_d', 'Stochastic %D'),
        ('vol_roll', 'Volatility (20d)')
    ]
    for col, name in tech_cols:
        if col in df_ready.columns:
            s = df_ready[col].dropna()
            if not s.empty:
                f.write(f"{name}: mean={s.mean():.6f}, std={s.std():.6f}, min={s.min():.6f}, max={s.max():.6f}\n")

    f.write("\n== Sentiment Analysis Summary ==\n")
    if 'sentiment_mean' in df_ready.columns:
        s = df_ready['sentiment_mean'].dropna()
        if not s.empty:
            f.write(f"Overall sentiment_mean: mean={s.mean():.6f}, std={s.std():.6f}\n")
    if 'news_count' in df_ready.columns:
        f.write(f"Total news_count: {int(df_ready['news_count'].fillna(0).sum())}\n")
    for cat in ['finance', 'general', 'international', 'national', 'policy', 'geopolitics']:
        sen_col = f"sentiment_{cat}"
        cnt_col = f"news_count_{cat}"
        if sen_col in df_ready.columns or cnt_col in df_ready.columns:
            mean_sen = df_ready[sen_col].dropna().mean() if sen_col in df_ready.columns else float('nan')
            total_cnt = int(df_ready[cnt_col].fillna(0).sum()) if cnt_col in df_ready.columns else 0
            f.write(f"{cat.title()}: avg_sentiment={mean_sen:.6f}, articles={total_cnt}\n")

    f.write("\n== Macroeconomic Indicators Summary ==\n")
    ret_series = df_ready['ret'].dropna() if 'ret' in df_ready.columns else pd.Series(dtype=float)
    for macro_col, pretty in [('GDP','GDP'),('CPI','CPI'),('Unemployment_Rate','Unemployment Rate')]:
        if macro_col in df_ready.columns:
            s = df_ready[macro_col].dropna()
            if not s.empty:
                corr = float('nan')
                if not ret_series.empty:
                    aligned = pd.concat([ret_series, s], axis=1).dropna()
                    if not aligned.empty:
                        corr = aligned.iloc[:,0].corr(aligned.iloc[:,1])
                f.write(f"{pretty}: range=({s.min():.6f} to {s.max():.6f}), corr_with_ret={corr:.6f}\n")

print("✅ Done. Outputs in", OUTDIR)
