"""
Technical Indicators Engine.
Provides clean and vectorized formulas for calculating standard technical analysis indicators on historical price data.
"""

import pandas as pd
import numpy as np


def add_sma(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.DataFrame:
    """
    Appends a Simple Moving Average (SMA) column to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        period (int): Lookback period. Defaults to 20.
        column (str): Price column to apply indicator on. Defaults to 'Close'.
        
    Returns:
        pd.DataFrame: DataFrame with the new SMA_{period} column.
    """
    df = df.copy()
    df[f"SMA_{period}"] = df[column].rolling(window=period).mean()
    return df


def add_ema(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.DataFrame:
    """
    Appends an Exponential Moving Average (EMA) column to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        period (int): Lookback decay window. Defaults to 20.
        column (str): Price column to apply indicator on. Defaults to 'Close'.
        
    Returns:
        pd.DataFrame: DataFrame with the new EMA_{period} column.
    """
    df = df.copy()
    df[f"EMA_{period}"] = df[column].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.DataFrame:
    """
    Appends a Relative Strength Index (RSI) column to the DataFrame.
    Calculated using standard Wilder's smoothing technique.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        period (int): Lookback window. Defaults to 14.
        column (str): Price column to apply RSI on. Defaults to 'Close'.
        
    Returns:
        pd.DataFrame: DataFrame with the new RSI_{period} column.
    """
    df = df.copy()
    delta = df[column].diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate wilder's rolling average using numpy arrays for performance and zero warnings
    avg_gain = gain.rolling(window=period).mean().values
    avg_loss = loss.rolling(window=period).mean().values
    
    gain_val = gain.values
    loss_val = loss.values
    
    for i in range(period, len(df)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain_val[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss_val[i]) / period
        
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = "Close"
) -> pd.DataFrame:
    """
    Appends MACD, Signal, and MACD Histogram columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        fast_period (int): Short EMA window. Defaults to 12.
        slow_period (int): Long EMA window. Defaults to 26.
        signal_period (int): Signal EMA window. Defaults to 9.
        column (str): Target price column. Defaults to 'Close'.
        
    Returns:
        pd.DataFrame: DataFrame with MACD, MACD_Signal, and MACD_Hist columns.
    """
    df = df.copy()
    fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
    
    df["MACD"] = fast_ema - slow_ema
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    column: str = "Close"
) -> pd.DataFrame:
    """
    Appends Bollinger Bands (Middle, Upper, and Lower bands) columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        period (int): Moving average window. Defaults to 20.
        num_std (float): Standard deviations multiplier. Defaults to 2.0.
        column (str): Target price column. Defaults to 'Close'.
        
    Returns:
        pd.DataFrame: DataFrame with BB_Middle, BB_Upper, and BB_Lower columns.
    """
    df = df.copy()
    df["BB_Middle"] = df[column].rolling(window=period).mean()
    rstd = df[column].rolling(window=period).std()
    
    df["BB_Upper"] = df["BB_Middle"] + (rstd * num_std)
    df["BB_Lower"] = df["BB_Middle"] - (rstd * num_std)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Appends an Average True Range (ATR) volatility measurement column to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame (requires 'High', 'Low', 'Close' columns).
        period (int): Smoothing window. Defaults to 14.
        
    Returns:
        pd.DataFrame: DataFrame with ATR_{period} column.
    """
    df = df.copy()
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    
    # True Range (TR) formula: Max of (H - L, |H - C_prev|, |L - C_prev|)
    close_prev = df["Close"].shift(1).values
    
    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)
    
    tr = np.max(np.column_stack([tr1, tr2, tr3]), axis=1)
    
    # Calculate rolling mean first
    tr_series = pd.Series(tr, index=df.index)
    atr = tr_series.rolling(window=period).mean().values
    
    # Apply Wilder's smoothing recursively
    for i in range(period, len(df)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
    df[f"ATR_{period}"] = atr
    return df
