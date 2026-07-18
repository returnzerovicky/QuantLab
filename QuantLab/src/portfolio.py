"""
Portfolio Analytics Module.
Allows allocating weights across multiple assets, constructing aggregate portfolio equity curves, and analyzing risk/return vs a benchmark index.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.data_loader import load_ticker_data
from src.metrics import generate_full_metrics_suite


def build_weighted_portfolio(
    tickers: List[str],
    weights: List[float],
    start_date: str,
    end_date: str,
    initial_capital: float = 10000.0,
    benchmark_ticker: str = "SPY"
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Simulates a multi-asset static-weight portfolio and compares it with a benchmark.
    
    Args:
        tickers (List[str]): List of ticker symbols.
        weights (List[float]): Weights allocated to each asset (must sum to 1.0).
        start_date (str): Backtest start date YYYY-MM-DD.
        end_date (str): Backtest end date YYYY-MM-DD.
        initial_capital (float): Beginning portfolio cash. Defaults to 10000.0.
        benchmark_ticker (str): Asset to compare against (e.g., 'SPY').
        
    Returns:
        Tuple:
            - pd.DataFrame: Combined daily returns, portfolio equity curve, and individual stock prices.
            - Dict[str, Any]: Quantitative metrics suite for the portfolio.
            - Dict[str, Any]: Quantitative metrics suite for the benchmark.
    """
    if len(tickers) != len(weights):
        raise ValueError("Tickers and weights lists must be of equal length.")
        
    if not np.isclose(sum(weights), 1.0):
        # Normalize weights if they don't sum to 1.0
        weights = [w / sum(weights) for w in weights]
        
    # Load all ticker datasets
    asset_data = {}
    common_dates: List[pd.DatetimeIndex] = []
    
    for t in tickers:
        df = load_ticker_data(t, start_date, end_date)
        asset_data[t] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        common_dates.append(df.index)
        
    # Load benchmark dataset
    try:
        benchmark_df = load_ticker_data(benchmark_ticker, start_date, end_date)
        benchmark_series = benchmark_df["Adj Close"] if "Adj Close" in benchmark_df.columns else benchmark_df["Close"]
    except Exception:
        # Fallback to single SPY mock or manual return array if download fails
        benchmark_series = None
        
    # Align dates across all series
    if not common_dates:
        raise ValueError("No data could be retrieved for any specified portfolio ticker.")
        
    # Intersect indexes
    aligned_index = common_dates[0]
    for idx in common_dates[1:]:
        aligned_index = aligned_index.intersection(idx)
        
    if benchmark_series is not None:
        aligned_index = aligned_index.intersection(benchmark_series.index)
        
    if len(aligned_index) < 10:
        raise ValueError("Insufficient date alignment overlap among asset histories.")
        
    # Build aligned price matrix
    prices_df = pd.DataFrame(index=aligned_index)
    returns_df = pd.DataFrame(index=aligned_index)
    
    for t in tickers:
        prices_df[t] = asset_data[t].loc[aligned_index]
        returns_df[t] = prices_df[t].pct_change().fillna(0.0)
        
    # Calculate weighted daily returns
    portfolio_daily_returns = returns_df.dot(weights)
    
    # Construct portfolio equity curve
    portfolio_equity = initial_capital * (1.0 + portfolio_daily_returns).cumprod()
    portfolio_equity.name = "Portfolio"
    
    # Construct benchmark equity curve
    if benchmark_series is not None:
        bench_prices = benchmark_series.loc[aligned_index]
        bench_returns = bench_prices.pct_change().fillna(0.0)
        benchmark_equity = initial_capital * (1.0 + bench_returns).cumprod()
        benchmark_equity.name = "Benchmark"
    else:
        benchmark_equity = portfolio_equity.copy() # dummy
        benchmark_equity.name = "Benchmark"
        bench_returns = portfolio_daily_returns.copy()
        
    # Compile output history DataFrame
    history_df = pd.DataFrame({
        "Portfolio_Return": portfolio_daily_returns,
        "Portfolio_Equity": portfolio_equity,
        "Benchmark_Return": bench_returns,
        "Benchmark_Equity": benchmark_equity
    }, index=aligned_index)
    
    # Add stock index weights and values
    for i, t in enumerate(tickers):
        history_df[f"{t}_Price"] = prices_df[t]
        # Stock-specific cumulative growth for plotting
        history_df[f"{t}_Growth"] = initial_capital * (prices_df[t] / prices_df[t].iloc[0])
        
    # Calculate full risk analytics
    port_suite = generate_full_metrics_suite(portfolio_equity, benchmark_equity)
    bench_suite = generate_full_metrics_suite(benchmark_equity, benchmark_equity) # self beta is 1.0, alpha 0.0
    
    # Calculate MPT statistics and correlation matrix
    mpt_results = optimize_portfolio_mpt(returns_df)
    mpt_results["diversification_ratio"] = calculate_diversification_ratio(weights, returns_df)
    
    return history_df, port_suite, bench_suite, mpt_results


def calculate_diversification_ratio(weights: List[float], returns_df: pd.DataFrame) -> float:
    """
    Calculates the Diversification Ratio of a portfolio.
    Formula: Weighted Average Asset Volatility / Portfolio Volatility
    """
    if len(weights) != len(returns_df.columns):
        return 1.0
    w = np.array(weights)
    vols = returns_df.std() * np.sqrt(252)
    weighted_vol = np.sum(w * vols)
    
    cov_matrix = returns_df.cov() * 252
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    if port_vol <= 0:
        return 1.0
    return float(weighted_vol / port_vol)


def optimize_portfolio_mpt(returns_df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Uses SLSQP optimization to compute Maximum Sharpe and Minimum Variance portfolios.
    Traces points on the Efficient Frontier.
    """
    import scipy.optimize as sco
    
    num_assets = len(returns_df.columns)
    if num_assets == 0:
        return {}
        
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    # Helper for portfolio statistics
    def portfolio_performance(weights):
        p_ret = np.dot(weights, mean_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        p_sharpe = (p_ret - risk_free_rate) / p_vol if p_vol > 0 else 0.0
        return p_ret, p_vol, p_sharpe

    # Negative Sharpe for maximization
    def min_func_sharpe(weights):
        return -portfolio_performance(weights)[2]

    # Portfolio Variance for minimization
    def min_func_variance(weights):
        return portfolio_performance(weights)[1] ** 2

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_weights = num_assets * [1.0 / num_assets]

    # Optimize Max Sharpe Portfolio
    opt_sharpe = sco.minimize(min_func_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    max_sharpe_weights = opt_sharpe.x.tolist() if opt_sharpe.success else init_weights
    sh_ret, sh_vol, sh_sr = portfolio_performance(max_sharpe_weights)

    # Optimize Min Variance Portfolio
    opt_var = sco.minimize(min_func_variance, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    min_var_weights = opt_var.x.tolist() if opt_var.success else init_weights
    mv_ret, mv_vol, mv_sr = portfolio_performance(min_var_weights)

    # Trace Efficient Frontier points
    min_ret = mv_ret
    max_ret = max(mean_returns)
    
    # Prevent edge cases with identical asset returns
    if min_ret >= max_ret:
        max_ret = min_ret + 0.1
        
    target_returns = np.linspace(min_ret, max_ret, 15)
    efficient_vols = []
    
    for target in target_returns:
        cons = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'eq', 'fun': lambda x: portfolio_performance(x)[0] - target}
        )
        opt_frontier = sco.minimize(min_func_variance, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
        if opt_frontier.success:
            efficient_vols.append(np.sqrt(opt_frontier.fun))
        else:
            efficient_vols.append(None)
            
    frontier_points = []
    for r, v in zip(target_returns, efficient_vols):
        if v is not None:
            frontier_points.append({"return": float(r), "volatility": float(v)})

    # Calculate correlation matrix and covariance matrix
    corr_matrix = returns_df.corr().to_dict()
    cov_dict = cov_matrix.to_dict()

    return {
        "max_sharpe": {
            "weights": dict(zip(returns_df.columns, max_sharpe_weights)),
            "return": float(sh_ret),
            "volatility": float(sh_vol),
            "sharpe": float(sh_sr)
        },
        "min_variance": {
            "weights": dict(zip(returns_df.columns, min_var_weights)),
            "return": float(mv_ret),
            "volatility": float(mv_vol),
            "sharpe": float(mv_sr)
        },
        "frontier": frontier_points,
        "correlation": corr_matrix,
        "covariance": cov_dict
    }
