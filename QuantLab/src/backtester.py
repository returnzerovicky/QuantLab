"""
Backtesting Engine.
Simulates historical performance of technical signals with transaction fees and execution lag to prevent look-ahead bias.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from src.metrics import generate_full_metrics_suite


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    commission_pct: float = 0.001,  # 0.1% transaction friction (slippage + broker fees)
    benchmark_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulates trading performance based on the 'Signal' column.
    
    Args:
        df (pd.DataFrame): Target DataFrame containing 'Close' and 'Signal' columns.
        initial_capital (float): Beginning portfolio cash.
        commission_pct (float): Static trading fee on trade size.
        benchmark_df (Optional[pd.DataFrame]): Optional dataframe containing SPY to calculate Alpha/Beta.
        
    Returns:
        Tuple:
            - pd.DataFrame: Backtest ledger containing portfolio equity curve, trades, returns.
            - Dict[str, Any]: Comprehensive metrics dictionary.
    """
    if "Signal" not in df.columns:
        raise ValueError("DataFrame must contain a 'Signal' column for backtesting.")
        
    backtest = df.copy()
    
    # Calculate asset daily returns
    backtest["Asset_Return"] = backtest["Close"].pct_change().fillna(0.0)
    
    # Apply 1-day lag to execution: yesterday's signal dictates today's position
    # This represents standard end-of-day execution models and prevents look-ahead bias.
    backtest["Position"] = backtest["Signal"].shift(1).fillna(0.0)
    
    # Calculate raw strategy daily returns before trading costs
    backtest["Strategy_Raw_Return"] = backtest["Position"] * backtest["Asset_Return"]
    
    # Determine trades (where position changes)
    # Position change size = |Today's position - Yesterday's position|
    backtest["Position_Change"] = backtest["Position"].diff().abs().fillna(0.0)
    
    # On row 0, if we start with a position, it counts as a trade
    if len(backtest) > 0 and backtest["Position"].iloc[0] != 0.0:
        backtest.at[backtest.index[0], "Position_Change"] = abs(backtest["Position"].iloc[0])
        
    # Trading costs = change size * closing price * fee factor? 
    # Or simpler: daily friction as a subtraction from returns based on position changes
    backtest["Transaction_Cost"] = backtest["Position_Change"] * commission_pct
    
    # Net daily return = raw return - transaction costs
    backtest["Strategy_Net_Return"] = backtest["Strategy_Raw_Return"] - backtest["Transaction_Cost"]
    
    # Build the strategy equity curve
    backtest["Equity_Curve"] = initial_capital * (1.0 + backtest["Strategy_Net_Return"]).cumprod()
    
    # Calculate Buy & Hold baseline equity curve
    backtest["Benchmark_Equity_Curve"] = initial_capital * (1.0 + backtest["Asset_Return"]).cumprod()
    
    # Calculate trade metrics
    trades = backtest[backtest["Position_Change"] > 0.0]
    num_trades = len(trades)
    
    # Calculate individual trade returns and win rate
    trade_profits = []
    win_rate = 0.0
    average_trade_return = 0.0
    
    # Reconstruct trades
    current_trade_return = 1.0
    in_trade = False
    
    for i in range(len(backtest)):
        pos = backtest["Position"].iloc[i]
        ret = backtest["Asset_Return"].iloc[i]
        
        # If in a position, accumulate returns
        if pos != 0.0:
            in_trade = True
            current_trade_return *= (1.0 + (pos * ret))
        else:
            if in_trade:
                # We exited the trade
                trade_profits.append(current_trade_return - 1.0)
                current_trade_return = 1.0
                in_trade = False
                
    # Append final trade if still in market
    if in_trade:
        trade_profits.append(current_trade_return - 1.0)
        
    if len(trade_profits) > 0:
        winning_trades = sum(1 for r in trade_profits if r > 0.0)
        win_rate = winning_trades / len(trade_profits)
        average_trade_return = np.mean(trade_profits)
        
    # Prepare Benchmark Series
    benchmark_equity = backtest["Benchmark_Equity_Curve"]
    if benchmark_df is not None:
        # Align index
        aligned_bench = benchmark_df.loc[benchmark_df.index.intersection(backtest.index)]
        if not aligned_bench.empty:
            benchmark_equity = initial_capital * (1.0 + aligned_bench["Close"].pct_change().fillna(0.0)).cumprod()
            benchmark_equity = benchmark_equity.reindex(backtest.index, method="ffill").fillna(initial_capital)
            
    # Calculate aggregate performance metrics
    metrics = generate_full_metrics_suite(backtest["Equity_Curve"], benchmark_equity)
    
    # Add strategy-specific trade metrics
    metrics.update({
        "initial_capital": initial_capital,
        "final_capital": float(backtest["Equity_Curve"].iloc[-1]),
        "total_net_return": float((backtest["Equity_Curve"].iloc[-1] / initial_capital) - 1.0),
        "benchmark_total_return": float((benchmark_equity.iloc[-1] / initial_capital) - 1.0),
        "total_trades": int(num_trades),
        "num_round_trips": len(trade_profits),
        "win_rate": float(win_rate),
        "avg_trade_return": float(average_trade_return),
        "total_commissions_paid": float((backtest["Transaction_Cost"] * initial_capital).sum())
    })
    
    return backtest, metrics
