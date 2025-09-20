# ============================
# File: hmm_model.py
# ============================
"""
hmm_model.py
- Fit GaussianHMM for regime detection
- Decode regimes and output posterior probabilities
- Save / load model (joblib)
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib
import os

def fit_hmm(df: pd.DataFrame, feature_cols: List[str], n_states: int = 3, cov_type: str = "full", random_state: int = 42) -> Tuple[GaussianHMM, StandardScaler]:
    """Fit HMM on training data only to avoid lookahead bias."""
    X = df[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = GaussianHMM(n_components=n_states, covariance_type=cov_type, n_iter=300, random_state=random_state)
    model.fit(Xs)
    return model, scaler

def fit_hmm_rolling(df: pd.DataFrame, feature_cols: List[str], n_states: int = 3, 
                   window_size: int = 252, min_periods: int = 100) -> pd.DataFrame:
    """
    Fit HMM using rolling windows for proper time series regime detection.
    This avoids lookahead bias by only using past data to determine current regime.
    """
    df_result = df.copy()
    df_result['regime'] = 0
    for i in range(n_states):
        df_result[f'prob_state_{i}'] = 0.0
    
    print(f"Fitting rolling HMM with window size {window_size}...")
    
    # Start loop from min_periods to ensure enough data for initial fit
    for i in range(min_periods, len(df)):
        # Use only past data for regime detection (data up to i-1)
        start_idx = max(0, i - window_size)
        train_data = df.iloc[start_idx:i][feature_cols].dropna()
        
        if len(train_data) < min_periods:
            continue
            
        try:
            # Fit HMM on past data (up to i-1)
            X = train_data.values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            
            model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
            model.fit(Xs)
            
            # Predict current regime using only past data
            current_features = df.iloc[i:i+1][feature_cols].values
            if not np.isnan(current_features).any():
                current_scaled = scaler.transform(current_features)
                post_probs = model.predict_proba(current_scaled)[0]
                regime = np.argmax(post_probs)
                
                df_result.iloc[i, df_result.columns.get_loc('regime')] = regime
                for j in range(n_states):
                    df_result.iloc[i, df_result.columns.get_loc(f'prob_state_{j}')] = post_probs[j]
                    
        except Exception as e:
            # If HMM fitting fails, use previous regime
            if i > 0:
                df_result.iloc[i, df_result.columns.get_loc('regime')] = df_result.iloc[i-1]['regime']
                for j in range(n_states):
                    df_result.iloc[i, df_result.columns.get_loc(f'prob_state_{j}')] = df_result.iloc[i-1][f'prob_state_{j}']
    
    print(f"✅ Rolling HMM completed. Regime distribution: {df_result['regime'].value_counts().to_dict()}")
    return df_result

def decode_hmm(model: GaussianHMM, scaler: StandardScaler, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    dfc = df.copy()
    Xs = scaler.transform(dfc[feature_cols].values)
    # posterior probabilities
    post = model.predict_proba(Xs)
    regimes = np.argmax(post, axis=1)
    dfc['regime'] = regimes
    for i in range(post.shape[1]):
        dfc[f'prob_state_{i}'] = post[:, i]
    return dfc

def save_hmm(model, scaler, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, path)

def load_hmm(path: str):
    obj = joblib.load(path)
    return obj['model'], obj['scaler']

if __name__ == "__main__":
    import data_pipeline as dp
    df = dp.fetch_price("^NSEI", "2016-01-01", "2024-01-01")
    feats = dp.compute_features(df)
    model, scaler = fit_hmm(feats, ['ret','vol_roll'], n_states=3)
    dec = decode_hmm(model, scaler, feats, ['ret','vol_roll'])
    print(dec[['regime','prob_state_0','prob_state_1','prob_state_2']].tail())
