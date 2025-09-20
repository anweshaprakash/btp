# ============================
# File: finbert_sentiment.py
# ============================
"""
FinBERT-based sentiment analysis for financial news.
Uses yiyanghkust/finbert-tone model for finance-specific sentiment scoring.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from typing import List, Dict, Union
import warnings
warnings.filterwarnings('ignore')

class FinBERTSentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial text."""
    
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone"):
        """Initialize FinBERT model and tokenizer."""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ FinBERT model loaded: {model_name}")
        except Exception as e:
            print(f"⚠️ FinBERT loading failed: {e}")
            print("Falling back to VADER sentiment...")
            self.model = None
            self.tokenizer = None
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text."""
        if self.model is None:
            return self._fallback_sentiment(text)
        
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # FinBERT labels: 0=negative, 1=neutral, 2=positive
            probs = probabilities.cpu().numpy()[0]
            
            return {
                'negative': float(probs[0]),
                'neutral': float(probs[1]), 
                'positive': float(probs[2]),
                'sentiment': float(probs[2] - probs[0]),  # positive - negative
                # ADDED: Include the probabilities separately for richer features
                'prob_negative': float(probs[0]),
                'prob_neutral': float(probs[1]),
                'prob_positive': float(probs[2])
            }
            
        except Exception as e:
            print(f"FinBERT analysis failed for text: {e}")
            return self._fallback_sentiment(text)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment of multiple texts."""
        if self.model is None:
            return [self._fallback_sentiment(text) for text in texts]
        
        try:
            # Batch tokenization
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Batch prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Process results
            results = []
            for probs in probabilities.cpu().numpy():
                results.append({
                    'negative': float(probs[0]),
                    'neutral': float(probs[1]),
                    'positive': float(probs[2]),
                    'sentiment': float(probs[2] - probs[0]),
                    # ADDED: Include the probabilities separately for richer features
                    'prob_negative': float(probs[0]),
                    'prob_neutral': float(probs[1]),
                    'prob_positive': float(probs[2])
                })
            
            return results
            
        except Exception as e:
            print(f"FinBERT batch analysis failed: {e}")
            return [self._fallback_sentiment(text) for text in texts]
    
    def _fallback_sentiment(self, text: str) -> Dict[str, float]:
        """Fallback to VADER sentiment if FinBERT fails."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            
            return {
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'positive': scores['pos'],
                'sentiment': scores['compound']
            }
        except:
            return {
                'negative': 0.0,
                'neutral': 1.0,
                'positive': 0.0,
                'sentiment': 0.0
            }

def compute_finbert_sentiment(df_news: pd.DataFrame, 
                            text_col: str = 'content',
                            max_text_length: int = 1000) -> pd.DataFrame:
    """
    Compute FinBERT sentiment for news DataFrame.
    
    Args:
        df_news: DataFrame with news articles
        text_col: Column containing text to analyze
        max_text_length: Maximum text length to process
    
    Returns:
        DataFrame with added sentiment columns
    """
    print("🔍 Computing FinBERT sentiment...")
    
    # Initialize analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    # Prepare texts
    texts = df_news[text_col].fillna('').astype(str)
    texts = texts.str[:max_text_length]  # Truncate long texts
    
    # Analyze in batches for efficiency
    batch_size = 32
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts.iloc[i:i+batch_size].tolist()
        batch_results = analyzer.analyze_batch(batch_texts)
        results.extend(batch_results)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} articles...")
    
    # Add results to DataFrame
    df_result = df_news.copy()
    df_result['finbert_negative'] = [r['negative'] for r in results]
    df_result['finbert_neutral'] = [r['neutral'] for r in results]
    df_result['finbert_positive'] = [r['positive'] for r in results]
    df_result['finbert_sentiment'] = [r['sentiment'] for r in results]
    
    print(f"✅ FinBERT sentiment computed for {len(df_result)} articles")
    print(f"Sentiment range: {df_result['finbert_sentiment'].min():.3f} to {df_result['finbert_sentiment'].max():.3f}")
    
    return df_result

if __name__ == "__main__":
    # Test the analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    test_texts = [
        "The company reported strong quarterly earnings, beating analyst expectations.",
        "Stock prices plummeted after the disappointing revenue guidance.",
        "Market volatility remains high amid economic uncertainty."
    ]
    
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result}")
        print()
