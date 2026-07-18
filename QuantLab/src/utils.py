"""
Quantitative Utility Helpers.
Provides statistical support functions, numerical cleaners, and scikit-learn lag features generators.
"""

from typing import Tuple, List, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import logging


def get_logger(name: str) -> logging.Logger:
    """
    Creates a standardized, beautifully configured logger for quantitative auditing.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def format_currency(val: float) -> str:
    """Formats float value into standard USD currency string."""
    return f"${val:,.2f}"


def format_percentage(val: float) -> str:
    """Formats decimal fractional return into structured percentage string."""
    return f"{val * 100.0:+.2f}%"


def clean_outliers(series: pd.Series, stdev_limit: float = 3.0) -> pd.Series:
    """
    Clips rolling distribution statistical outliers to mitigate systemic shocks.
    Useful for sanitizing daily return spikes.
    """
    mean = series.mean()
    std = series.std()
    lower = mean - (stdev_limit * std)
    upper = mean + (stdev_limit * std)
    return series.clip(lower, upper)


def create_ml_lag_features(
    df: pd.DataFrame,
    lags: Union[int, List[int]] = 5,
    target_col: str = "Close",
    include_pct_changes: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Helper function utilizing Scikit-Learn logic.
    Transforms raw OHLC data into lag features for machine learning time-series forecasting.
    
    Args:
        df (pd.DataFrame): Core price DataFrame.
        lags (Union[int, List[int]]): Number of lag steps or specific list of intervals.
        target_col (str): The column we want to predict (e.g., 'Close' or 'Daily_Return').
        include_pct_changes (bool): If True, computes lag percentages of the target.
        
    Returns:
        Tuple:
            - pd.DataFrame (X): Feature matrix.
            - pd.Series (y): Target series shifted forward by 1 step.
    """
    df_feat = pd.DataFrame(index=df.index)
    
    # Define active lag list
    lag_list = list(range(1, lags + 1)) if isinstance(lags, int) else lags
    
    for lag in lag_list:
        df_feat[f"{target_col}_Lag_{lag}"] = df[target_col].shift(lag)
        if include_pct_changes:
            df_feat[f"{target_col}_Lag_Pct_{lag}"] = df[target_col].pct_change(lag).shift(1)
            
    # Include some trailing moving standard deviation for volatility tracking
    df_feat["Rolling_Volatility_5"] = df[target_col].pct_change().rolling(window=5).std().shift(1)
    df_feat["Rolling_Volatility_20"] = df[target_col].pct_change().rolling(window=20).std().shift(1)
    
    # Target column is the next period Close / Return (shifted by -1)
    y = df[target_col].shift(-1)
    
    # Align and drop NaNs
    combined = pd.concat([df_feat, y], axis=1).dropna()
    X = combined.iloc[:, :-1]
    y_aligned = combined.iloc[:, -1]
    
    return X, y_aligned


def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Performs walk-forward cross-validation splits on time-series inputs using scikit-learn.
    
    Returns a list of (train_indices, test_indices) tuples.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X))
