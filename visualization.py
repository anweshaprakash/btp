"""
visualization.py

Plot:
- Price chart with colored regimes
- Regime probabilities over time
- Prediction vs actual plot
"""

from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_price_with_regimes(df: pd.DataFrame, date_col: str = None, close_col: str = "Close", regime_col: str = "regime", out_path: str = None):
    """
    Plot price and color background by regime.
    df must contain columns: Close and regime
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    times = df.index
    ax.plot(times, df[close_col], lw=1.2, label='Close')

    # For each contiguous block of regime, color the background
    regimes = df[regime_col].values
    unique_regimes = np.unique(regimes)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_regimes)))
    regime_color_map = {r: colors[i] for i, r in enumerate(unique_regimes)}
    # fill background
    start = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[i - 1]:
            r = regimes[i - 1]
            ax.axvspan(times[start], times[i - 1], color=regime_color_map[r], alpha=0.12)
            start = i
    ax.set_title("Price with HMM Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
    return fig


def plot_regime_probabilities(df: pd.DataFrame, prob_prefix: str = "prob_state_", out_path: str = None):
    """
    Plot regime probabilities over time (stacked or separate).
    """
    prob_cols = [c for c in df.columns if c.startswith(prob_prefix)]
    if not prob_cols:
        raise ValueError("No regime prob columns found with prefix " + prob_prefix)
    fig, ax = plt.subplots(figsize=(14, 4))
    for c in prob_cols:
        ax.plot(df.index, df[c], label=c)
    ax.set_title("HMM Regime Probabilities Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.legend()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
    return fig


def plot_predictions_vs_actual(dates, actuals, preds, out_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, actuals, label='Actual', lw=1)
    ax.plot(dates, preds, label='Predicted', lw=1)
    ax.set_title("Next-day Return: Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
    return fig
