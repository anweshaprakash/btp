# ============================
# File: data_pipeline.py
# ============================
"""
data_pipeline.py
- fetch_price: fetch OHLCV using yfinance with retries
- save/load CSV helpers
- compute_features: returns, rolling volatility, technical indicators
"""

from typing import Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import time
import os

def fetch_price(ticker: str, start: str, end: str, interval: str = "1d", max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch OHLCV price data with retry logic using yf.download.
    Returns DataFrame with columns: Open, High, Low, Close, Volume and Datetime index.
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if df is None or df.empty:
                raise ValueError(f"No data returned for {ticker} (attempt {attempt})")
            
            df.index = pd.to_datetime(df.index)
            
            # Flatten MultiIndex columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure columns exist
            expected = ['Open','High','Low','Close','Volume']
            for c in expected:
                if c not in df.columns:
                    df[c] = pd.NA
            return df[expected].copy()
        except Exception as e:
            print(f"[fetch_price] attempt {attempt} failed for {ticker}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch price for {ticker} after {max_retries} attempts")

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)

def compute_features(df: pd.DataFrame, vol_window: int = 20, ma_window: int = 10) -> pd.DataFrame:
    """
    Compute:
      - log returns "ret"
      - rolling volatility "vol_roll" (annualized)
      - rolling mean of returns "ret_ma"
      - several technical indicators (SMA, EMA, RSI)
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    if 'Close' not in df.columns:
        raise ValueError(f"Close column not found. Available columns: {df.columns.tolist()}")
    
    out = df.copy()
    out['ret'] = np.log(out['Close']).diff()
    
    # Only drop NaN if the 'ret' column exists and has data
    if 'ret' in out.columns and not out['ret'].isna().all():
        out = out.dropna(subset=['ret']).copy()
    else:
        print("Warning: No valid returns data found, skipping NaN removal")
    out['vol_roll'] = out['ret'].rolling(window=vol_window, min_periods=1).std() * np.sqrt(252)
    out['ret_ma'] = out['ret'].rolling(window=ma_window, min_periods=1).mean()
    out['sma'] = out['Close'].rolling(window=ma_window, min_periods=1).mean()
    out['ema'] = out['Close'].ewm(span=ma_window, adjust=False).mean()
    # RSI (simple implementation)
    delta = out['Close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    out['rsi'] = 100 - (100 / (1 + rs))

    # ADDED: MACD
    exp1 = out['Close'].ewm(span=12, adjust=False).mean()
    exp2 = out['Close'].ewm(span=26, adjust=False).mean()
    out['macd'] = exp1 - exp2
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    out['macd_hist'] = out['macd'] - out['macd_signal']

    # ADDED: Bollinger Bands
    out['bollinger_upper'] = out['sma'] + (out['Close'].rolling(window=20).std() * 2)
    out['bollinger_lower'] = out['sma'] - (out['Close'].rolling(window=20).std() * 2)
    
    # ADDED: Stochastic Oscillator
    low_14 = out['Low'].rolling(window=14).min()
    high_14 = out['High'].rolling(window=14).max()
    out['stoch_k'] = 100 * ((out['Close'] - low_14) / (high_14 - low_14 + 1e-9))
    out['stoch_d'] = out['stoch_k'].rolling(window=3).mean()

    out = out.ffill().bfill()
    return out

if __name__ == "__main__":
    df = fetch_price("RELIANCE.NS", "2018-01-01", "2024-12-31")
    feats = compute_features(df)
    print(feats.tail())
