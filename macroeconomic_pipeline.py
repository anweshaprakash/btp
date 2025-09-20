# ============================
# File: macroeconomic_pipeline.py
# ============================

import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import logging
import os

def fetch_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches key macroeconomic indicators from the FRED database.
    
    Returns a DataFrame with a daily DateTimeIndex, with values for 
    monthly/quarterly data points forward-filled.
    """
    
    # FRED series IDs for key macroeconomic indicators
    # These IDs are standard and can be found on the FRED website.
    fred_series = {
        'GDP': 'GDPC1',               # Real Gross Domestic Product (Quarterly)
        'CPI': 'CPIAUCSL',            # Consumer Price Index (Monthly)
        'Unemployment_Rate': 'UNRATE' # Unemployment Rate (Monthly)
    }
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"Fetching macroeconomic data from FRED...")
    
    try:
        # Fetch data for all series IDs at once
        macro_df = web.DataReader(list(fred_series.values()), 'fred', start_dt, end_dt)
        
        # Rename columns to be more descriptive
        macro_df.columns = fred_series.keys()
        
        # Resample to daily frequency and forward-fill missing values
        macro_df = macro_df.asfreq('D')
        macro_df = macro_df.fillna(method='ffill')
        
        # Fill any remaining NaNs at the beginning of the series with backward fill
        macro_df = macro_df.fillna(method='bfill')
        
        print("✅ Macroeconomic data fetched and processed successfully.")
        return macro_df
        
    except Exception as e:
        logging.error(f"Failed to fetch macroeconomic data: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    # Example usage
    macro_data = fetch_macro_data(start_date="2010-01-01", end_date="2025-09-18")
    if not macro_data.empty:
        print("\nFirst 5 rows of macroeconomic data:")
        print(macro_data.head())
        print("\nLast 5 rows of macroeconomic data:")
        print(macro_data.tail())