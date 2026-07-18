"""
Trading Strategies Module.
Contains signal generation logic for technical algorithmic trading strategies.
Signals:
    1.0  = LONG / BUY
   -1.0  = SHORT / SELL
    0.0  = LIQUIDATE / FLAT / OUT
"""

import pandas as pd
from src.indicators import add_sma, add_ema, add_rsi, add_macd


def buy_and_hold_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates a baseline Buy & Hold strategy.
    Signal is 1.0 (Long) on every row.
    """
    df = df.copy()
    df["Signal"] = 1.0
    return df


def sma_crossover_strategy(
    df: pd.DataFrame,
    fast_period: int = 50,
    slow_period: int = 200
) -> pd.DataFrame:
    """
    Two-period Simple Moving Average crossover strategy.
    
    Signal:
        Long (1.0) when Fast SMA is above Slow SMA.
        Flat or Short (0.0 or -1.0) when Fast SMA is below Slow SMA.
    """
    df = df.copy()
    df = add_sma(df, fast_period, "Close")
    df = add_sma(df, slow_period, "Close")
    
    fast_col = f"SMA_{fast_period}"
    slow_col = f"SMA_{slow_period}"
    
    # Generate binary position signals (1 when Fast > Slow, else 0)
    df["Signal"] = 0.0
    df.loc[df[fast_col] > df[slow_col], "Signal"] = 1.0
    
    return df


def ema_crossover_strategy(
    df: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21
) -> pd.DataFrame:
    """
    Exponential Moving Average crossover strategy.
    
    Signal:
        Long (1.0) when Fast EMA is above Slow EMA.
        Flat (0.0) when Fast EMA is below Slow EMA.
    """
    df = df.copy()
    df = add_ema(df, fast_period, "Close")
    df = add_ema(df, slow_period, "Close")
    
    fast_col = f"EMA_{fast_period}"
    slow_col = f"EMA_{slow_period}"
    
    df["Signal"] = 0.0
    df.loc[df[fast_col] > df[slow_col], "Signal"] = 1.0
    
    return df


def rsi_strategy(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0
) -> pd.DataFrame:
    """
    RSI trading strategy.
    
    Signal:
        Long (1.0) when RSI drops below the oversold threshold (rebound trigger).
        Flat (0.0) when RSI rises above the overbought threshold.
        Maintains current state in intermediate ranges.
    """
    df = df.copy()
    df = add_rsi(df, period, "Close")
    rsi_col = f"RSI_{period}"
    
    df["Signal"] = 0.0
    current_signal = 0.0
    
    for idx, row in df.iterrows():
        rsi_val = row[rsi_col]
        if pd.isna(rsi_val):
            continue
            
        if rsi_val < oversold:
            current_signal = 1.0  # Buy/Long
        elif rsi_val > overbought:
            current_signal = 0.0  # Sell/Liquidate
            
        df.at[idx, "Signal"] = current_signal
        
    return df


def macd_strategy(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    MACD Signal Line crossover strategy.
    
    Signal:
        Long (1.0) when MACD is above the MACD Signal line.
        Flat (0.0) when MACD is below the MACD Signal line.
    """
    df = df.copy()
    df = add_macd(df, fast_period, slow_period, signal_period, "Close")
    
    df["Signal"] = 0.0
    df.loc[df["MACD"] > df["MACD_Signal"], "Signal"] = 1.0
    
    return df
