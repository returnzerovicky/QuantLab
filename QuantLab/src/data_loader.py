"""
Data Loader Module.
Retrieves and validates historical market data from Yahoo Finance.
"""

from typing import Optional
import os
import pandas as pd
import yfinance as yf


def load_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    cache_dir: Optional[str] = "data"
) -> pd.DataFrame:
    """
    Downloads historical stock/cryptocurrency data from Yahoo Finance.
    Saves downloaded data as a CSV file inside the cache directory if provided.
    
    Args:
        ticker (str): The ticker symbol to fetch (e.g., 'AAPL', 'BTC-USD').
        start_date (str): The start date in format 'YYYY-MM-DD'.
        end_date (str): The end date in format 'YYYY-MM-DD'.
        interval (str): Data resolution ('1d', '1wk', '1mo', '1h', etc.). Defaults to '1d'.
        cache_dir (Optional[str]): Path to data caching folder. If None, caching is disabled.
        
    Returns:
        pd.DataFrame: A formatted Pandas DataFrame containing Open, High, Low, Close, Adj Close, Volume.
    """
    ticker = ticker.strip().upper()
    
    # Check if cached file exists and spans the requested range
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Clean ticker symbol for filename
        clean_ticker = "".join(c for c in ticker if c.isalnum() or c in "-_")
        cache_path = os.path.join(cache_dir, f"{clean_ticker}_{start_date}_{end_date}_{interval}.csv")
        
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
            except Exception:
                pass # Fall back to download if cache read fails
                
    try:
        # Download using yfinance
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}' from {start_date} to {end_date}.")
            
        # Standardize MultiIndex column names if returned by yfinance (recent versions return multi-index if single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Standardize index and name it Date
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        
        # Verify required columns exist
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required market column: {col}")
                
        # Fill missing values
        df = df.ffill().bfill()
        
        # Save to cache if enabled
        if cache_path:
            df.to_csv(cache_path)
            
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch market data for {ticker}: {str(e)}")


def validate_ticker(ticker: str) -> bool:
    """
    Quick ticker verification using yfinance metadata.
    
    Args:
        ticker (str): The ticker symbol to check.
        
    Returns:
        bool: True if ticker is valid and exists, False otherwise.
    """
    ticker_obj = yf.Ticker(ticker)
    try:
        # Fetching history is more reliable than the ticker.info dictionary
        hist = ticker_obj.history(period="1d")
        return not hist.empty
    except Exception:
        return False
