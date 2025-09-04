"""
hmm_model.py

Train/evaluate a Gaussian HMM for regime detection using hmmlearn.
- fit_hmm: fits a GaussianHMM to multivariate features
- decode_regimes: apply to DataFrame and return regime labels + probabilities
"""

from typing import Tuple
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib


def fit_hmm(df_features: pd.DataFrame, feature_cols: list, n_states: int = 3, covariance_type: str = "full", random_state: int = 42) -> Tuple[GaussianHMM, StandardScaler]:
    """
    Fit a GaussianHMM on selected features.

    Args:
        df_features: DataFrame with features
        feature_cols: list of columns to use in HMM
        n_states: number of hidden regimes
        covariance_type: 'full', 'diag'
    Returns:
        fitted HMM, fitted StandardScaler
    """
    X = df_features[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=200, random_state=random_state)
    model.fit(Xs)
    return model, scaler


def decode_regimes(model: GaussianHMM, scaler: StandardScaler, df_features: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Decode regimes and append columns for regime label and regime probabilities.

    Returns:
        DataFrame (copy) with additional columns:
            - regime (int)
            - prob_state_0 ... prob_state_{n-1}
    """
    df = df_features.copy()
    Xs = scaler.transform(df[feature_cols].values)
    # posterior probabilities for each time
    post = model.predict_proba(Xs)
    regimes = np.argmax(post, axis=1)
    df['regime'] = regimes
    # add probs columns
    for i in range(post.shape[1]):
        df[f'prob_state_{i}'] = post[:, i]
    return df


def save_model(model: GaussianHMM, scaler: StandardScaler, path_model: str):
    joblib.dump({'model': model, 'scaler': scaler}, path_model)


def load_model(path_model: str):
    obj = joblib.load(path_model)
    return obj['model'], obj['scaler']


if __name__ == "__main__":
    import data_pipeline as dp
    df = dp.fetch_data("SPY", "2015-01-01", "2024-01-01")
    feats = dp.compute_features(df)
    feature_cols = ['ret', 'vol_roll']
    model, scaler = fit_hmm(feats, feature_cols)
    decoded = decode_regimes(model, scaler, feats, feature_cols)
    print(decoded[['ret', 'vol_roll', 'regime']].tail())
