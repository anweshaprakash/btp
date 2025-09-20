# ============================
# File: visualization.py
# ============================
"""
Visualization utilities for the financial sentiment analysis pipeline.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Optional
import os

def plot_predictions(dates: List[pd.Timestamp], true_values: np.ndarray, 
                    predicted_values: np.ndarray, out_path: str, 
                    title: str = "Predictions vs Actual") -> None:
    """
    Plot predictions vs actual values over time.
    
    Args:
        dates: List of dates corresponding to the data points
        true_values: Actual values
        predicted_values: Predicted values
        out_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted
    plt.plot(dates, true_values, label='Actual', alpha=0.7, linewidth=2)
    plt.plot(dates, predicted_values, label='Predicted', alpha=0.7, linewidth=2)
    
    # Add scatter points for better visibility
    plt.scatter(dates, true_values, alpha=0.5, s=20, label='Actual Points')
    plt.scatter(dates, predicted_values, alpha=0.5, s=20, label='Predicted Points')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_regime_probabilities(dates: List[pd.Timestamp], regime_probs: np.ndarray, 
                            out_path: str, n_states: int = 3) -> None:
    """
    Plot HMM regime probabilities over time.
    
    Args:
        dates: List of dates
        regime_probs: Array of regime probabilities (n_samples, n_states)
        out_path: Path to save the plot
        n_states: Number of regimes
    """
    plt.figure(figsize=(14, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:n_states]
    labels = [f'Regime {i}' for i in range(n_states)]
    
    # Stack plot for regime probabilities
    plt.stackplot(dates, *regime_probs.T, colors=colors, alpha=0.7, labels=labels)
    
    plt.title('HMM Regime Probabilities Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_price_regimes(dates: List[pd.Timestamp], prices: np.ndarray, 
                      regimes: np.ndarray, out_path: str, 
                      title: str = "Price with HMM Regimes") -> None:
    """
    Plot price data colored by HMM regimes.
    
    Args:
        dates: List of dates
        prices: Price data
        regimes: Regime assignments
        out_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Create color map for regimes
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot price line
    plt.plot(dates, prices, color='black', linewidth=1, alpha=0.8, label='Price')
    
    # Color background by regime
    for i, (date, price) in enumerate(zip(dates, prices)):
        if i < len(regimes):
            regime = regimes[i]
            color = colors[regime % len(colors)]
            plt.axvline(x=date, color=color, alpha=0.1, linewidth=0.5)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sentiment_analysis(dates: List[pd.Timestamp], sentiment_data: dict, 
                           out_path: str, title: str = "Sentiment Analysis") -> None:
    """
    Plot various sentiment metrics over time.
    
    Args:
        dates: List of dates
        sentiment_data: Dictionary with sentiment metrics
        out_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(14, 10))
    
    n_metrics = len(sentiment_data)
    colors = plt.cm.Set3(np.linspace(0, 1, n_metrics))
    
    for i, (metric_name, values) in enumerate(sentiment_data.items()):
        plt.subplot(n_metrics, 1, i+1)
        plt.plot(dates, values, color=colors[i], linewidth=2, label=metric_name)
        plt.title(f'{metric_name}', fontsize=12, fontweight='bold')
        plt.ylabel('Value', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if i == n_metrics - 1:  # Last subplot
            plt.xlabel('Date', fontsize=12)
        else:
            plt.xticks([])
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
