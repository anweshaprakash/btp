# BTP: Regime-Aware Time Series Forecasting (HMM + LSTM)

This project implements a full pipeline for financial time-series modeling that combines Hidden Markov Models (HMM) for market regime detection with LSTM-based regression for next-day return prediction. It fetches OHLCV data via Yahoo Finance, extracts features, trains an HMM to infer regimes and probabilities, and trains two neural models: a baseline LSTM and a hybrid LSTM that incorporates HMM regime probabilities. The pipeline produces evaluation metrics and plots under `outputs/`.

## Features
- **Data fetching**: Download OHLCV data from Yahoo Finance.
- **Feature engineering**: Log-returns, rolling volatility, moving-average of returns, optional normalization.
- **Regime detection**: Gaussian HMM on engineered features; outputs regime labels and per-state probabilities.
- **Forecasting models**:
  - Baseline LSTM using past returns.
  - Hybrid LSTM using past returns + HMM regime probabilities.
- **Visualization**: Price colored by regimes, regime probabilities over time, and predictions vs actuals.
- **Reporting**: Saves metrics and a summary report.

## Repository Structure
- `main.py`: Orchestrates the end-to-end pipeline.
- `data_pipeline.py`: Fetches data and computes features.
- `hmm_model.py`: Trains/decodes Gaussian HMM, save/load utilities.
- `lstm_model.py`: Dataset, model, train/evaluate routines for LSTM.
- `visualization.py`: Matplotlib plotting utilities.
- `outputs/`: Generated figures, trained models, and a text report.
- `requirements.txt`: Python package dependencies.

## Installation
1. Create a Python 3.10+ environment (recommended):
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start
Run the full pipeline (downloads data, trains HMM + LSTMs, saves outputs):
```bash
python main.py
```
Outputs will be saved under `outputs/`.

## Configuration
Adjust the top of `main.py` to change data source, training parameters, and output locations:
- **Ticker and dates**:
  - `TICKER` (e.g., `"SPY"`, `"AAPL"`, `"BTC-USD"`)
  - `START`, `END` (YYYY-MM-DD)
- **Features and models**:
  - `VOL_WINDOW`: rolling volatility window (days)
  - `HMM_FEATURES`: columns used by HMM (default: `['ret', 'vol_roll']`)
  - `N_STATES`: number of HMM regimes
  - `SEQ_LEN`: LSTM input sequence length
  - `BATCH_SIZE`, `EPOCHS`, `DEVICE` ("cpu" or "cuda")
- **Outputs**:
  - `OUTPUT_DIR`, `FIG_DIR`, `MODEL_DIR`

## Data and Features
From Yahoo Finance OHLCV, the pipeline derives:
- `ret`: log return of `Close`
- `vol_roll`: rolling standard deviation of `ret` (annualized)
- `ret_ma`: simple moving average of returns
Optional z-score normalization can be enabled in `compute_features(..., normalize=True)`.

## Models
- **HMM (hmmlearn.GaussianHMM)**
  - Trained on `HMM_FEATURES`
  - Produces `regime` labels and per-state probabilities `prob_state_0..k`
  - Saved to `outputs/models/hmm_{TICKER}.pkl`
- **LSTM (PyTorch)**
  - Baseline inputs: `['ret']` (modifiable)
  - Hybrid inputs: baseline + `prob_state_*`
  - Best checkpoints saved to `outputs/models/lstm_base.pth` and `outputs/models/lstm_hybrid.pth`

## Train/Validate/Test Split
`main.py` performs a chronological split: 70% train, 15% validation, 15% test. `TimeSeriesDataset` uses sliding windows of length `SEQ_LEN` and predicts next-day return.

## Outputs
Generated artifacts under `outputs/`:
- `figures/price_regimes.png`: Price with regime shading.
- `figures/regime_probs.png`: Regime probabilities over time.
- `figures/pred_base.png`: Baseline LSTM predictions vs actuals.
- `figures/pred_hybrid.png`: Hybrid LSTM predictions vs actuals.
- `models/hmm_{TICKER}.pkl`: Trained HMM + scaler.
- `models/lstm_base.pth`, `models/lstm_hybrid.pth`: Trained LSTM weights.
- `report.txt`: Summary including RMSEs and a proxy regime accuracy metric.

## Reproducibility and Tips
- Ensure stable internet access for Yahoo Finance downloads.
- If CUDA is available, set `DEVICE = "cuda"` in `main.py` for faster training.
- Add `ret_ma` or `vol_roll` to `baseline_inputs` in `main.py` to enrich baseline features.
- For different horizons, modify the dataset target or sequence construction in `lstm_model.py`.

## Troubleshooting
- "No data found": Check `TICKER` symbol and date range in `main.py`.
- Empty/NaN features: Ensure sufficient history; `compute_features` drops the first return and forward/backward fills small gaps in `main.py`.
- Matplotlib display issues: This project saves figures to disk; a display backend is not required.
- CUDA errors: Fall back to CPU by setting `DEVICE = "cpu"`.
