# ============================
# File: timeseries_lstm.py
# ============================
"""
Time series LSTM with HMM regime context.
Uses regime information to condition the LSTM predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd

class RegimeConditionedLSTM(nn.Module):
    """
    LSTM that uses HMM regime information as conditioning context.
    """
    
    def __init__(self, 
                 input_dim: int,
                 regime_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_dim: int = 1):
        super(RegimeConditionedLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.regime_dim = regime_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Main LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Regime conditioning layers
        self.regime_encoder = nn.Sequential(
            nn.Linear(regime_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Attention mechanism for regime context
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # LSTM + regime context
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, regime_context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with regime conditioning.
        
        Args:
            x: Input sequences [batch_size, seq_len, input_dim]
            regime_context: Regime probabilities [batch_size, regime_dim]
            
        Returns:
            Predictions: [batch_size, output_dim]
        """
        # Process input through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        
        # Encode regime context
        regime_encoded = self.regime_encoder(regime_context)  # [batch_size, hidden_dim]
        regime_encoded = regime_encoded.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply attention between LSTM output and regime context
        attended_out, _ = self.attention(
            query=lstm_out,
            key=regime_encoded,
            value=regime_encoded
        )
        
        # Take last timestep
        last_output = attended_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Combine LSTM output with regime context
        combined = torch.cat([last_output, regime_encoded.squeeze(1)], dim=1)
        
        # Generate final output
        output = self.output_layers(combined)
        
        return output

class TimeSeriesLSTMDataset(DataLoader):
    """Dataset for time series LSTM with regime context."""
    
    def __init__(self, 
                 df: 'pd.DataFrame',
                 features: List[str],
                 regime_features: List[str],
                 target_col: str,
                 seq_len: int = 20):
        self.df = df
        self.features = features
        self.regime_features = regime_features
        self.target_col = target_col
        self.seq_len = seq_len
        
        # Prepare data
        self.feature_data = df[features].values
        self.regime_data = df[regime_features].values
        self.targets = df[target_col].values
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Create sequences for LSTM training."""
        sequences = []
        
        # Start at seq_len to ensure full sequence is available
        # Stop at len(data)-1 because we need target at i+1
        for i in range(self.seq_len, len(self.feature_data) - 1):
            # Feature sequence (from i-seq_len to i-1)
            feature_seq = torch.FloatTensor(self.feature_data[i-self.seq_len:i])
            
            # Regime context (from day i-1)
            # FIX: This is the critical change to fix the lookahead bug
            regime_context = torch.FloatTensor(self.regime_data[i-1])
            
            # Target (from day i)
            target = torch.FloatTensor([self.targets[i]])
            
            sequences.append((feature_seq, regime_context, target))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def train_regime_lstm(model: RegimeConditionedLSTM,
                     train_loader: 'DataLoader',
                     val_loader: 'DataLoader',
                     n_epochs: int = 20,
                     lr: float = 1e-3,
                     device: str = 'cpu',
                     save_path: str = None) -> RegimeConditionedLSTM:
    """Train regime-conditioned LSTM model."""
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for feature_batch, regime_batch, target_batch in train_loader:
            feature_batch = feature_batch.to(device)
            regime_batch = regime_batch.to(device)
            target_batch = target_batch.to(device)
            
            outputs = model(feature_batch, regime_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for feature_batch, regime_batch, target_batch in val_loader:
                feature_batch = feature_batch.to(device)
                regime_batch = regime_batch.to(device)
                target_batch = target_batch.to(device)
                
                outputs = model(feature_batch, regime_batch)
                loss = criterion(outputs, target_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
    
    return model

if __name__ == "__main__":
    # Test the regime-conditioned LSTM
    batch_size = 32
    seq_len = 20
    input_dim = 10
    regime_dim = 3
    hidden_dim = 64
    
    # Create dummy data
    feature_inputs = torch.randn(batch_size, seq_len, input_dim)
    regime_context = torch.randn(batch_size, regime_dim)
    
    # Create model
    model = RegimeConditionedLSTM(
        input_dim=input_dim,
        regime_dim=regime_dim,
        hidden_dim=hidden_dim
    )
    
    # Forward pass
    output = model(feature_inputs, regime_context)
    print(f"Input shapes: features={feature_inputs.shape}, regime={regime_context.shape}")
    print(f"Output shape: {output.shape}")