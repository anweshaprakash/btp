# BTP: Financial Time Series Regime-Switching Model

This project implements a comprehensive pipeline for financial time-series modeling that combines multiple data sources and advanced machine learning techniques. The system integrates:

- **Price Data**: OHLCV data from Yahoo Finance with technical indicators
- **News Sentiment**: Multi-source news collection with FinBERT sentiment analysis
- **Macroeconomic Data**: Key economic indicators from FRED database
- **Regime Detection**: Hidden Markov Models (HMM) for market regime identification
- **Advanced Forecasting**: Multiple LSTM variants including regime-conditioned models

The pipeline produces comprehensive evaluation metrics, visualizations, and trained models under `outputs/`.

## Features
- **Multi-Source Data Collection**:
  - OHLCV price data from Yahoo Finance with technical indicators (RSI, MACD, Bollinger Bands, Stochastic Oscillator)
  - News sentiment analysis using FinBERT and VADER fallback
  - Macroeconomic indicators from FRED (GDP, CPI, Unemployment Rate)
- **Advanced Feature Engineering**:
  - Technical indicators and rolling statistics
  - News sentiment aggregation by category and time windows
  - Macroeconomic data alignment and forward-filling
  - Rich sentiment features including momentum and volatility measures
- **Regime Detection**: Rolling Gaussian HMM for market regime identification with proper time-series handling
- **Multiple Forecasting Models**:
  - Baseline LSTM using technical indicators
  - Hybrid LSTM incorporating news sentiment and macroeconomic data
  - Regime-Conditioned LSTM with attention mechanisms
- **Comprehensive Visualization**: Price regimes, sentiment analysis, predictions vs actuals
- **Robust Evaluation**: Multiple model comparison with RMSE metrics and statistical analysis

## Repository Structure
- `main.py`: Main orchestration script that runs the complete pipeline
- `data_pipeline.py`: Price data fetching and technical indicator computation
- `news_pipeline.py`: Multi-source news collection and categorization
- `finbert_sentiment.py`: FinBERT-based sentiment analysis for financial text
- `macroeconomic_pipeline.py`: FRED database integration for economic indicators
- `feature_engineering.py`: Data alignment and advanced feature creation
- `hmm_model.py`: Rolling HMM implementation for regime detection
- `lstm_model.py`: Standard LSTM models and evaluation utilities
- `timeseries_lstm.py`: Regime-conditioned LSTM with attention mechanisms
- `visualization.py`: Comprehensive plotting utilities for all data types
- `outputs/`: Generated figures, trained models, and evaluation reports
- `requirements.txt`: Complete Python package dependencies

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
The main configuration is in `main.py` under the `config` dictionary. Key parameters:

### Data Sources
- **`TICKER`**: Stock symbol (e.g., `"RELIANCE.NS"`, `"SPY"`, `"AAPL"`)
- **`START`, `END`**: Date range in YYYY-MM-DD format
- **News Sources**: Google News RSS, GDELT API (automatic fallback)

### Model Parameters
- **`VOL_WINDOW`**: Rolling volatility window (default: 20 days)
- **`N_STATES`**: Number of HMM regimes (default: 3)
- **`SEQ_LEN`**: LSTM input sequence length (default: 20)
- **`BATCH`**: Training batch size (default: 32)
- **`EPOCHS`**: Training epochs (default: 20)
- **`LEARNING_RATE`**: Adam optimizer learning rate (default: 1e-3)
- **`HIDDEN_DIM`**: LSTM hidden dimension (default: 64)
- **`NUM_LAYERS`**: LSTM layers (default: 2)
- **`DEVICE`**: Training device ("cpu" or "cuda")

### Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic Oscillator
- **Sentiment Features**: Category-specific sentiment, momentum, volatility measures
- **Macroeconomic Data**: GDP, CPI, Unemployment Rate (from FRED)

### Output Configuration
- **`OUTDIR`**: Main output directory (default: "outputs")
- **`FIGDIR`**: Figures directory (default: "outputs/figures")
- **`MODELDIR`**: Models directory (default: "outputs/models")

## Data and Features

### Price Data Features
From Yahoo Finance OHLCV data:
- **`ret`**: Log return of Close price
- **`vol_roll`**: Rolling standard deviation of returns (annualized)
- **`ret_ma`**: Simple moving average of returns
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic Oscillator

### News Sentiment Features
From multi-source news collection:
- **`sentiment_mean`**: Daily average sentiment score
- **`news_count`**: Daily article count
- **Category-specific sentiment**: Finance, general, international, national, policy, geopolitics
- **Rich sentiment features**: Momentum, volatility, intensity measures
- **FinBERT probabilities**: Negative, neutral, positive probability scores

### Macroeconomic Features
From FRED database:
- **`GDP`**: Real Gross Domestic Product (quarterly, forward-filled)
- **`CPI`**: Consumer Price Index (monthly, forward-filled)
- **`Unemployment_Rate`**: Unemployment Rate (monthly, forward-filled)

### Feature Engineering
- Automatic forward/backward filling for missing values
- Z-score normalization for all features
- Time-aligned data aggregation across all sources

## Models

### Regime Detection
- **Rolling HMM (hmmlearn.GaussianHMM)**
  - Uses rolling windows to avoid lookahead bias
  - Trained on technical features: `['ret', 'vol_roll']`
  - Produces regime labels and state probabilities
  - Implements proper time-series regime detection

### Sentiment Analysis
- **FinBERT (yiyanghkust/finbert-tone)**
  - Finance-specific sentiment analysis
  - Provides negative, neutral, positive probabilities
  - Fallback to VADER sentiment if FinBERT fails
  - Batch processing for efficiency

### Forecasting Models
- **Baseline LSTM (PyTorch)**
  - Input: Technical indicators only
  - Features: RSI, MACD, Bollinger Bands, Stochastic Oscillator
  - Architecture: Multi-layer LSTM with dropout

- **Hybrid LSTM (PyTorch)**
  - Input: Technical indicators + news sentiment + macroeconomic data
  - Features: All baseline features plus sentiment and macro indicators
  - Enhanced with rich sentiment features and category-specific analysis

- **Regime-Conditioned LSTM (PyTorch)**
  - Advanced architecture with attention mechanisms
  - Uses HMM regime probabilities as conditioning context
  - Multi-head attention between LSTM output and regime context
  - Separate regime encoder and output layers

### Model Persistence
- HMM models: `outputs/models/hmm.pkl`
- LSTM models: `outputs/models/lstm_base.pth`, `lstm_hyb.pth`
- Regime-conditioned: `outputs/models/timeseries_lstm_regime.pth`

## Train/Validate/Test Split
`main.py` performs a chronological split: 70% train, 15% validation, 15% test. `TimeSeriesDataset` uses sliding windows of length `SEQ_LEN` and predicts next-day return.

## Outputs
Generated artifacts under `outputs/`:

### Visualizations (`outputs/figures/`)
- **`price_regimes.png`**: Price chart with HMM regime shading
- **`regime_probs.png`**: HMM regime probabilities over time
- **`pred_base.png`**: Baseline LSTM predictions vs actual returns
- **`pred_hyb.png`**: Hybrid LSTM predictions vs actual returns

### Trained Models (`outputs/models/`)
- **`hmm.pkl`**: Trained HMM model with scaler
- **`lstm_base.pth`**: Baseline LSTM model weights
- **`lstm_hyb.pth`**: Hybrid LSTM model weights
- **`timeseries_lstm_regime.pth`**: Regime-conditioned LSTM weights
- **`lstm_vol_base.pth`**: Volatility-focused LSTM weights
- **`lstm_vol_hybrid.pth`**: Hybrid volatility LSTM weights
- **`timeseries_lstm_jump.pth`**: Jump detection LSTM weights

### Reports (`outputs/`)
- **`report.txt`**: Comprehensive evaluation report including:
  - Model performance metrics (RMSE for all models)
  - Statistical analysis results
  - Sentiment regression analysis
  - Data quality summaries

## Reproducibility and Tips
- **Internet Access**: Ensure stable connection for Yahoo Finance, FRED, and news API downloads
- **GPU Acceleration**: Set `DEVICE = "cuda"` in `main.py` for faster training (if available)
- **Feature Engineering**: Modify `baseline_inputs` and `hybrid_inputs` in `main.py` to experiment with different feature combinations
- **News Sources**: The pipeline automatically falls back between Google News RSS and GDELT if one fails
- **Sentiment Analysis**: FinBERT provides finance-specific sentiment; VADER is used as fallback
- **Time Horizons**: Modify sequence construction in `lstm_model.py` for different prediction horizons

## Troubleshooting
- **"No data found"**: Check `TICKER` symbol and date range in `main.py`
- **Empty/NaN features**: Ensure sufficient historical data; the pipeline handles missing values automatically
- **News collection failures**: The system gracefully handles API failures and continues with available data
- **FinBERT loading issues**: Automatic fallback to VADER sentiment analysis
- **CUDA errors**: Set `DEVICE = "cpu"` in configuration
- **Memory issues**: Reduce `BATCH` size or `SEQ_LEN` for large datasets
- **FRED API limits**: Macroeconomic data fetching may be rate-limited; the pipeline handles this gracefully
