"""
Financial Metrics Module.
Calculates risk-adjusted returns, drawdown ratios, and regression statistics (Alpha and Beta) relative to a benchmark.
"""

from typing import Dict, Union, Optional
import numpy as np
import pandas as pd


def daily_returns(series: pd.Series) -> pd.Series:
    """
    Computes standard arithmetic daily percent changes.
    """
    return series.pct_change().fillna(0.0)


def log_returns(series: pd.Series) -> pd.Series:
    """
    Computes logarithmic daily returns.
    """
    return np.log(series / series.shift(1)).fillna(0.0)


def cagr(series: pd.Series, periods_per_year: float = 252.0) -> float:
    """
    Calculates Compound Annual Growth Rate (CAGR).
    
    Formula: (End Value / Start Value) ^ (1 / Years) - 1
    """
    if len(series) < 2 or series.iloc[0] <= 0:
        return 0.0
    
    total_returns = series.iloc[-1] / series.iloc[0]
    years = len(series) / periods_per_year
    
    if years <= 0:
        return 0.0
    
    return float((total_returns ** (1 / years)) - 1.0)


def volatility(returns_series: pd.Series, periods_per_year: float = 252.0) -> float:
    """
    Calculates annualized volatility of daily returns.
    """
    if len(returns_series) < 2:
        return 0.0
    return float(returns_series.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns_series: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: float = 252.0
) -> float:
    """
    Calculates the Sharpe Ratio (annualized).
    
    Formula: (Annualized Return - Risk Free Rate) / Annualized Volatility
    """
    if len(returns_series) < 2:
        return 0.0
        
    ann_ret = np.mean(returns_series) * periods_per_year
    ann_vol = volatility(returns_series, periods_per_year)
    
    if ann_vol <= 0:
        return 0.0
        
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns_series: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: float = 252.0
) -> float:
    """
    Calculates the Sortino Ratio (annualized).
    Focuses only on downside deviations.
    """
    if len(returns_series) < 2:
        return 0.0
        
    ann_ret = np.mean(returns_series) * periods_per_year
    
    # Isolate downside returns
    downside_returns = returns_series[returns_series < 0]
    if len(downside_returns) < 2:
        return 0.0
        
    # Annualized downside deviation
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_std <= 0:
        return 0.0
        
    return float((ann_ret - risk_free_rate) / downside_std)


def max_drawdown(series: pd.Series) -> Dict[str, Union[float, pd.Timestamp]]:
    """
    Calculates the Maximum Peak-to-Trough Drawdown.
    
    Returns:
        Dict: Maximum Drawdown percentage, peak date, and trough date.
    """
    if len(series) < 1:
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None}
        
    roll_max = series.cummax()
    drawdowns = (series - roll_max) / roll_max
    max_dd = drawdowns.min()
    
    trough_date = drawdowns.idxmin()
    # Find the peak date leading up to the trough
    pre_trough_series = series.loc[:trough_date]
    peak_date = pre_trough_series.idxmax() if not pre_trough_series.empty else None
    
    return {
        "max_drawdown": float(max_dd),
        "peak_date": peak_date,
        "trough_date": trough_date
    }


def alpha_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: float = 252.0
) -> Dict[str, float]:
    """
    Calculates Alpha and Beta (relative to a benchmark index) using linear regression.
    
    Formula:
        Return_asset = Alpha + Beta * Return_benchmark + error
    """
    # Align the series indexes
    df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    df.columns = ["asset", "benchmark"]
    
    if len(df) < 5:
        return {"alpha": 0.0, "beta": 1.0}
    
    covariance = df["asset"].cov(df["benchmark"])
    benchmark_variance = df["benchmark"].var()
    
    if benchmark_variance <= 0:
        beta = 1.0
    else:
        beta = float(covariance / benchmark_variance)
        
    # Annualized rates
    asset_ann_ret = df["asset"].mean() * periods_per_year
    benchmark_ann_ret = df["benchmark"].mean() * periods_per_year
    
    # CAPM Alpha formula
    alpha = float((asset_ann_ret - risk_free_rate) - beta * (benchmark_ann_ret - risk_free_rate))
    
    return {"alpha": alpha, "beta": beta}


def value_at_risk_historical(returns_series: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculates Historical Value at Risk (VaR).
    Returns VaR as a positive fraction (e.g. 0.02 means 2% daily loss limit).
    """
    if len(returns_series) < 5:
        return 0.0
    percentile = (1.0 - confidence_level) * 100
    return -float(np.percentile(returns_series, percentile))


def value_at_risk_parametric(returns_series: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculates Parametric (Variance-Covariance) Value at Risk (VaR).
    Assumes standard normal distribution.
    """
    if len(returns_series) < 5:
        return 0.0
    mean = returns_series.mean()
    std = returns_series.std()
    
    # Z-scores for common confidence levels
    if np.isclose(confidence_level, 0.95):
        z = 1.64485
    elif np.isclose(confidence_level, 0.99):
        z = 2.32635
    elif np.isclose(confidence_level, 0.90):
        z = 1.28155
    else:
        try:
            from scipy.stats import norm
            z = norm.ppf(confidence_level)
        except ImportError:
            # Fallback interpolation for Z
            z = 1.64485 + (confidence_level - 0.95) * (2.32635 - 1.64485) / 0.04
            
    return -float(mean - z * std)


def conditional_value_at_risk(returns_series: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculates Conditional Value at Risk (CVaR / Expected Shortfall).
    Mean of returns that exceed the Historical VaR threshold.
    """
    if len(returns_series) < 5:
        return 0.0
    var_cutoff = -value_at_risk_historical(returns_series, confidence_level)
    tail_returns = returns_series[returns_series <= var_cutoff]
    if len(tail_returns) == 0:
        return value_at_risk_historical(returns_series, confidence_level)
    return -float(tail_returns.mean())


def calmar_ratio(series: pd.Series, periods_per_year: float = 252.0) -> float:
    """
    Calculates Calmar Ratio.
    Formula: CAGR / Max Drawdown
    """
    ann_cagr = cagr(series, periods_per_year)
    max_dd_val = abs(max_drawdown(series)["max_drawdown"])
    if max_dd_val <= 0:
        return 0.0
    return float(ann_cagr / max_dd_val)


def information_ratio(returns_series: pd.Series, benchmark_returns: pd.Series, periods_per_year: float = 252.0) -> float:
    """
    Calculates Information Ratio.
    Formula: Annualized Active Return / Annualized Tracking Error
    """
    # Align the series indexes
    df = pd.concat([returns_series, benchmark_returns], axis=1).dropna()
    df.columns = ["strategy", "benchmark"]
    if len(df) < 5:
        return 0.0
    
    active_returns = df["strategy"] - df["benchmark"]
    ann_active_ret = active_returns.mean() * periods_per_year
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
    
    if tracking_error <= 0:
        return 0.0
    return float(ann_active_ret / tracking_error)


def treynor_ratio(returns_series: pd.Series, beta: float, risk_free_rate: float = 0.02, periods_per_year: float = 252.0) -> float:
    """
    Calculates Treynor Ratio.
    Formula: (Annualized Return - Risk Free Rate) / Beta
    """
    if beta == 0:
        return 0.0
    ann_ret = returns_series.mean() * periods_per_year
    return float((ann_ret - risk_free_rate) / beta)


def profit_factor(returns_series: pd.Series) -> float:
    """
    Calculates Profit Factor.
    Formula: Sum of positive returns / Absolute sum of negative returns
    """
    pos_returns = returns_series[returns_series > 0]
    neg_returns = returns_series[returns_series < 0]
    sum_pos = pos_returns.sum()
    sum_neg = abs(neg_returns.sum())
    if sum_neg <= 0:
        return 0.0 if sum_pos <= 0 else 999.0
    return float(sum_pos / sum_neg)


def rolling_volatility(returns_series: pd.Series, window: int = 20, periods_per_year: float = 252.0) -> pd.Series:
    """
    Calculates Rolling Annualized Volatility.
    """
    return returns_series.rolling(window).std() * np.sqrt(periods_per_year)


def rolling_sharpe_ratio(returns_series: pd.Series, window: int = 20, risk_free_rate: float = 0.02, periods_per_year: float = 252.0) -> pd.Series:
    """
    Calculates Rolling Annualized Sharpe Ratio.
    """
    roll_mean = returns_series.rolling(window).mean() * periods_per_year
    roll_std = returns_series.rolling(window).std() * np.sqrt(periods_per_year)
    # Avoid division by zero
    return (roll_mean - risk_free_rate) / roll_std.replace(0, np.nan)


def generate_full_metrics_suite(
    equity_series: pd.Series,
    benchmark_series: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, Union[float, str, None]]:
    """
    Aggregates all risk and return metrics for a strategy portfolio series.
    """
    returns = daily_returns(equity_series)
    
    ann_cagr = cagr(equity_series)
    ann_vol = volatility(returns)
    sharpe = sharpe_ratio(returns, risk_free_rate)
    sortino = sortino_ratio(returns, risk_free_rate)
    dd_stats = max_drawdown(equity_series)
    
    var_hist = value_at_risk_historical(returns, 0.95)
    var_param = value_at_risk_parametric(returns, 0.95)
    cvar = conditional_value_at_risk(returns, 0.95)
    calmar = calmar_ratio(equity_series)
    prof_fact = profit_factor(returns)
    
    metrics = {
        "cagr": ann_cagr,
        "volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": dd_stats["max_drawdown"],
        "max_drawdown_peak": str(dd_stats["peak_date"].date()) if dd_stats["peak_date"] else None,
        "max_drawdown_trough": str(dd_stats["trough_date"].date()) if dd_stats["trough_date"] else None,
        "var_historical": var_hist,
        "var_parametric": var_param,
        "cvar": cvar,
        "calmar_ratio": calmar,
        "profit_factor": prof_fact,
        "alpha": 0.0,
        "beta": 1.0,
        "treynor_ratio": 0.0,
        "information_ratio": 0.0
    }
    
    if benchmark_series is not None:
        benchmark_rets = daily_returns(benchmark_series)
        ab = alpha_beta(returns, benchmark_rets, risk_free_rate)
        metrics["alpha"] = ab["alpha"]
        metrics["beta"] = ab["beta"]
        metrics["treynor_ratio"] = treynor_ratio(returns, ab["beta"], risk_free_rate)
        metrics["information_ratio"] = information_ratio(returns, benchmark_rets)
        
    return metrics
