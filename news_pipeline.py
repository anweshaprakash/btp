# ============================
# File: news_pipeline.py
# ============================
"""
news_pipeline.py
- Collects news from multiple free sources (Google News RSS, GDELT, scraping)
- Classifies into categories (finance, national, international, policy, geopolitics)
- Adds sentiment scoring (VADER fallback)
- Returns DataFrame with: published_at, source, title, content, category, sentiment, url
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from textblob import TextBlob
import logging

# Optional: VADER sentiment
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    vader = SentimentIntensityAnalyzer()
except Exception:
    vader = None


# ----------------------------
# Google News RSS Fetcher
# ----------------------------
def fetch_news_google(query: str, max_articles: int = 50) -> pd.DataFrame:
    """
    Fetch news using Google News RSS (free).
    """
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.content, "xml")
    items = soup.find_all("item")[:max_articles]

    rows = []
    for it in items:
        rows.append({
            "published_at": pd.to_datetime(it.pubDate.text) if it.pubDate else datetime.utcnow(),
            "source": it.source.text if it.source else "GoogleNews",
            "title": it.title.text if it.title else "",
            "content": it.description.text if it.description else "",
            "url": it.link.text if it.link else ""
        })
    return pd.DataFrame(rows)


# ----------------------------
# GDELT Fetcher
# ----------------------------
def fetch_news_gdelt(query: str, days: int = 1, max_articles: int = 50) -> pd.DataFrame:
    """
    Fetch recent global news from GDELT 2.0 API.
    """
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_articles),
        "format": "json"
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        data = r.json()
        rows = []
        for art in data.get("articles", []):
            rows.append({
                "published_at": pd.to_datetime(art.get("seendate")),
                "source": art.get("source"),
                "title": art.get("title"),
                "content": art.get("seendate") or "",
                "url": art.get("url")
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logging.warning(f"GDELT fetch failed: {e}")
        return pd.DataFrame(columns=["published_at", "source", "title", "content", "url"])


# ----------------------------
# Categorization
# ----------------------------
def categorize_article(title: str, content: str) -> str:
    """
    Simple rule-based categorizer.
    """
    text = f"{title} {content}".lower()
    if any(k in text for k in ["stock", "market", "finance", "profit", "loss", "nifty", "sensex"]):
        return "finance"
    elif any(k in text for k in ["budget", "policy", "government", "regulation"]):
        return "policy"
    elif any(k in text for k in ["india", "delhi", "mumbai", "bangalore"]):
        return "national"
    elif any(k in text for k in ["us", "china", "russia", "ukraine", "global", "world"]):
        return "international"
    elif any(k in text for k in ["war", "geopolitics", "conflict", "oil", "trade"]):
        return "geopolitics"
    else:
        return "general"

def is_high_impact_news(title: str, content: str, company_name: str = "RELIANCE") -> bool:
    """Filter for high-impact news only."""
    text = f"{title} {content}".lower()
    company_lower = company_name.lower()
    
    # High-impact indicators
    high_impact_keywords = [
        # Company-specific
        f'{company_lower}', 'reliance', 'ril',
        
        # Financial events
        'earnings', 'quarterly results', 'annual results', 'guidance', 'forecast',
        'revenue', 'profit', 'loss', 'beats', 'misses', 'exceeds', 'falls short',
        
        # Market events
        'stock split', 'dividend', 'buyback', 'merger', 'acquisition', 'deal',
        'upgrade', 'downgrade', 'price target', 'analyst', 'rating',
        
        # Sector events
        'oil', 'petroleum', 'refinery', 'chemical', 'telecom', 'jio', 'retail',
        'energy', 'renewable', 'solar', 'wind',
        
        # Regulatory events
        'regulatory', 'approval', 'license', 'permit', 'clearance', 'policy',
        'government', 'ministry', 'budget', 'tax', 'subsidy',
        
        # Major announcements
        'announces', 'launches', 'expands', 'invests', 'partnership', 'joint venture',
        'ceo', 'chairman', 'management', 'leadership'
    ]
    
    # Check for high-impact keywords
    has_high_impact = any(keyword in text for keyword in high_impact_keywords)
    
    # Additional filters
    is_recent = True  # Assume all fetched news is recent
    has_substance = len(content) > 100  # Avoid very short articles
    
    return has_high_impact and is_recent and has_substance


# ----------------------------
# Sentiment Scoring
# ----------------------------
def compute_sentiment(text: str) -> float:
    if not text:
        return 0.0
    if vader:
        return vader.polarity_scores(text)["compound"]
    else:
        return TextBlob(text).sentiment.polarity


# ----------------------------
# High-level Aggregator
# ----------------------------
def fetch_news_combined(query: str, max_articles: int = 100) -> pd.DataFrame:
    """
    Fetch from Google News + GDELT, filter for high-impact news, and use FinBERT sentiment.
    """
    dfs = []
    try:
        dfs.append(fetch_news_google(query, max_articles=max_articles))
    except Exception as e:
        logging.warning(f"Google News fetch failed: {e}")
    try:
        dfs.append(fetch_news_gdelt(query, max_articles=max_articles))
    except Exception as e:
        logging.warning(f"GDELT fetch failed: {e}")

    if not dfs:
        return pd.DataFrame(columns=["published_at", "source", "title", "content", "category", "sentiment", "url"])

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["url"])
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce").fillna(pd.Timestamp.utcnow())

    # Filter for high-impact news only
    print(f"🔍 Filtering for high-impact news from {len(df)} articles...")
    high_impact_mask = df.apply(
        lambda row: is_high_impact_news(row['title'], row['content'], query), axis=1
    )
    df = df[high_impact_mask].copy()
    print(f"✅ High-impact news: {len(df)} articles")

    if df.empty:
        return pd.DataFrame(columns=["published_at", "source", "title", "content", "category", "sentiment", "url"])

    # Categorize articles
    df["category"] = df.apply(lambda x: categorize_article(x["title"], x["content"]), axis=1)
    
    # Use FinBERT for sentiment analysis
    try:
        from finbert_sentiment import compute_finbert_sentiment
        df = compute_finbert_sentiment(df, text_col='content')
        # Use FinBERT sentiment as primary sentiment score
        df["sentiment"] = df["finbert_sentiment"]
        print("✅ FinBERT sentiment analysis completed")
    except Exception as e:
        print(f"⚠️ FinBERT failed, using VADER: {e}")
        # Fallback to VADER
        df["sentiment"] = df["content"].apply(compute_sentiment)

    return df.reset_index(drop=True)