# ============================
# File: feature_engineering.py
# ============================
"""
feature_engineering.py

Functions:
- align_price_news: aggregate news (overall + per-category) into daily features and join with price data
- align_macro_data: join macroeconomic data with the primary DataFrame
- dataset_target_dates: helper to map dataset sample indices to target dates (used by main/visualization)
"""

from typing import List
import pandas as pd
import numpy as np


def align_price_news(df_price: pd.DataFrame, df_news: pd.DataFrame, date_col: str = "published_at") -> pd.DataFrame:
    """
    Align price data with news sentiment aggregated per day and per category.
    """
    dfp = df_price.copy()
    dfn = df_news.copy()
    if date_col not in dfn.columns:
        raise ValueError(f"news dataframe missing '{date_col}' column")

    dfn['published_at'] = pd.to_datetime(dfn['published_at'], utc=True, errors='coerce').fillna(pd.Timestamp.utcnow())
    dfn['published_at'] = dfn['published_at'].dt.tz_localize(None)
    dfn['date'] = dfn['published_at'].dt.normalize()

    daily = dfn.groupby('date').agg(
        news_count=('sentiment', 'count'),
        sentiment_mean=('sentiment', 'mean')
    )

    prob_cols = [c for c in dfn.columns if c.startswith('finbert_')]
    if prob_cols:
        daily_probs = dfn.groupby('date')[prob_cols].mean()
    else:
        daily_probs = pd.DataFrame(index=daily.index)

    if 'category' in dfn.columns:
        cat = dfn.groupby(['date', 'category'])['sentiment'].mean().unstack(fill_value=np.nan)
        cat_count = dfn.groupby(['date', 'category'])['sentiment'].count().unstack(fill_value=0)

        cat.columns = [f"sentiment_{str(c).strip().lower().replace(' ', '_')}" for c in cat.columns]
        cat_count.columns = [f"news_count_{str(c).strip().lower().replace(' ', '_')}" for c in cat_count.columns]
        cat = pd.concat([cat, cat_count], axis=1)
    else:
        cat = pd.DataFrame(index=daily.index)

    agg = pd.concat([daily, cat, daily_probs], axis=1).sort_index()
    agg = agg.fillna({'news_count': 0, 'sentiment_mean': 0}).fillna(0)

    dfp = dfp.copy()
    dfp['_date'] = dfp.index.normalize()
    
    if agg.index.tz is not None:
        agg.index = agg.index.tz_localize(None)
    if dfp['_date'].dt.tz is not None:
        dfp['_date'] = dfp['_date'].dt.tz_localize(None)

    merged = dfp.merge(agg, left_on='_date', right_index=True, how='left')

    merged['sentiment_mean'] = merged['sentiment_mean'].fillna(0.0)
    merged['news_count'] = merged['news_count'].fillna(0.0)

    for col in agg.columns:
        if col not in merged.columns:
            merged[col] = 0.0
    
    merged['sentiment_pos_count_3d'] = (merged['sentiment_mean'] > 0.1).rolling(3, min_periods=1).sum()
    merged['sentiment_neg_count_3d'] = (merged['sentiment_mean'] < -0.1).rolling(3, min_periods=1).sum()
    merged['sentiment_neutral_count_3d'] = ((merged['sentiment_mean'] >= -0.1) & (merged['sentiment_mean'] <= 0.1)).rolling(3, min_periods=1).sum()
    
    merged['sentiment_vol_3d'] = merged['sentiment_mean'].rolling(3, min_periods=1).std()
    merged['sentiment_vol_7d'] = merged['sentiment_mean'].rolling(7, min_periods=1).std()
    
    merged['sentiment_momentum_3d'] = merged['sentiment_mean'].rolling(3, min_periods=1).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0)
    
    merged['news_intensity_3d'] = merged['news_count'].rolling(3, min_periods=1).mean()
    merged['news_intensity_7d'] = merged['news_count'].rolling(7, min_periods=1).mean()
    
    for cat in ['finance', 'general', 'international', 'national']:
        if f'sentiment_{cat}' in merged.columns:
            merged[f'sentiment_{cat}_strength'] = merged[f'sentiment_{cat}'].abs()
            merged[f'news_{cat}_ratio'] = merged[f'news_count_{cat}'] / (merged['news_count'] + 1e-6)
            merged[f'sentiment_{cat}_vol_3d'] = merged[f'sentiment_{cat}'].rolling(3, min_periods=1).std()

    merged['sentiment_vol_interaction'] = merged['sentiment_mean'] * merged['vol_roll']
    if 'finbert_prob_positive' in merged.columns:
        merged['pos_vol_interaction'] = merged['finbert_prob_positive'] * merged['vol_roll']
    if 'finbert_prob_negative' in merged.columns:
        merged['neg_vol_interaction'] = merged['finbert_prob_negative'] * merged['vol_roll']

    merged = merged.drop(columns=['_date'])
    merged = merged.ffill().bfill().fillna(0.0)

    return merged

def align_macro_data(df_base: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merges a macroeconomic DataFrame (daily, filled) into the main DataFrame.
    """
    # Ensure both DataFrames are timezone-naive
    if df_base.index.tz is not None:
        df_base.index = df_base.index.tz_localize(None)
    if df_macro.index.tz is not None:
        df_macro.index = df_macro.index.tz_localize(None)

    # Use a left join to align macro data with the primary time series index
    aligned_df = df_base.merge(df_macro, how='left', left_index=True, right_index=True)

    # Forward-fill any gaps that might appear from the merge
    aligned_df = aligned_df.ffill().bfill().fillna(0)

    print("✅ Macroeconomic data aligned with main price data.")
    return aligned_df


def dataset_target_dates(df_slice: pd.DataFrame, seq_len: int) -> List[pd.Timestamp]:
    dates = list(df_slice.index)
    target_dates = []
    for end in range(seq_len - 1, len(dates) - 1):
        target_dates.append(dates[end + 1])
    return target_dates


def make_basic_features(df_price: pd.DataFrame, df_news: pd.DataFrame) -> pd.DataFrame:
    return align_price_news(df_price, df_news)