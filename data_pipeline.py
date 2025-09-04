"""
data_pipeline.py

Functions to fetch and preprocess financial price data using yfinance:
- fetch_data: download OHLCV
- compute_features: returns, rolling volatility, optional normalization
"""

from typing import Tuple
import pandas as pd
import numpy as np
import yfinance as yf


def fetch_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Args:
        ticker: e.g. "AAPL" or "BTC-USD"
        start: "YYYY-MM-DD"
        end: "YYYY-MM-DD"
        interval: e.g. "1d"

    Returns:
        DataFrame indexed by Datetime with columns: Open, High, Low, Close, Volume
    """
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(start=start, end=end, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker} between {start} and {end}")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)
    return df


def compute_features(df: pd.DataFrame, vol_window: int = 20, normalize: bool = False) -> pd.DataFrame:
    """
    Compute features:
    - log returns (daily)
    - rolling volatility (std of returns)
    - optional normalization (z-score)

    Args:
        df: price DataFrame with 'Close' column
        vol_window: window length for rolling volatility (days)
        normalize: if True, z-score normalize features across the dataset

    Returns:
        DataFrame with columns: Close, ret, vol_roll, (normalized features if requested)
    """
    out = df.copy()
    out['ret'] = np.log(out['Close']).diff()  # log returns
    # drop first NaN
    out = out.dropna(subset=['ret']).copy()
    out['vol_roll'] = out['ret'].rolling(window=vol_window, min_periods=1).std() * np.sqrt(252)  # annualized
    # simple moving average of returns (optional)
    out['ret_ma'] = out['ret'].rolling(window=vol_window, min_periods=1).mean()

    feature_cols = ['ret', 'vol_roll', 'ret_ma']
    if normalize:
        for c in feature_cols:
            mean_c = out[c].mean()
            std_c = out[c].std(ddof=0) if out[c].std(ddof=0) > 0 else 1.0
            out[c + '_z'] = (out[c] - mean_c) / std_c
    return out


if __name__ == "__main__":
    # quick test
    df = fetch_data("AAPL", "2020-01-01", "2024-01-01")
    feats = compute_features(df, vol_window=20)
    print(feats.tail())
