/**
 * QuantLab File Registry.
 * Holds the exact, complete production code of the Python repository
 * to display in the UI's code explorer and generate the ZIP download.
 */

export interface CodeFile {
  path: string;
  name: string;
  language: string;
  content: string;
}

export const QUANTLAB_FILES: Record<string, CodeFile> = {
  "app.py": {
    path: "app.py",
    name: "app.py",
    language: "python",
    content: `\"\"\"
QuantLab Streamlit Application.
The primary interactive user dashboard for historical market analysis, multi-asset portfolio management,
technical indicators, and algorithmic strategy backtesting.
\"\"\"

import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from src.data_loader import load_ticker_data, validate_ticker
from src.indicators import add_sma, add_ema, add_rsi, add_macd, add_bollinger_bands, add_atr
from src.strategies import sma_crossover_strategy, ema_crossover_strategy, rsi_strategy, macd_strategy
from src.backtester import run_backtest
from src.portfolio import build_weighted_portfolio
from src.visualizer import (
    plot_candlestick_with_indicators,
    plot_backtest_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_portfolio_allocation
)
from src.utils import create_ml_lag_features, format_currency, format_percentage

# Configure Streamlit page options
st.set_page_config(
    page_title="QuantLab | Quantitative Analytics & Backtesting",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render customized elegant CSS
st.markdown(\"\"\"
<style>
    .reportview-container {
        background: #0f172a;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #10b981 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
</style>
\"\"\", unsafe_allow_html=True)

# Main Banner Header
st.title("QuantLab 🔬📊")
st.markdown("---")

# Navigation Sidebar
st.sidebar.title("🎛️ Analytics Panel")
app_mode = st.sidebar.selectbox(
    "Choose Lab Module",
    ["1. Market Data & Indicators", "2. Strategy Backtester", "3. Portfolio Allocator", "4. Machine Learning Sandbox"]
)

# Default constants
DEFAULT_START = datetime.date(2023, 1, 1)
DEFAULT_END = datetime.date(2026, 1, 1)

# ==========================================================
# MODULE 1: MARKET DATA & INDICATORS
# ==========================================================
if app_mode == "1. Market Data & Indicators":
    st.header("📈 Technical Indicators & Market Explorer")
    
    # Grid columns for configurations
    cfg_col1, cfg_col2, cfg_col3 = st.columns([1.5, 1, 1])
    
    with cfg_col1:
        ticker = st.text_input("Ticker Symbol (e.g., AAPL, BTC-USD, MSFT, GLD)", "AAPL").strip().upper()
    with cfg_col2:
        start_date = st.date_input("Start Date", DEFAULT_START)
    with cfg_col3:
        end_date = st.date_input("End Date", DEFAULT_END)
        
    intervals = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    selected_int = st.sidebar.selectbox("Data Interval", list(intervals.keys()))
    
    # List of technical overlays
    st.sidebar.subheader("Overlay Indicators")
    sma_active = st.sidebar.checkbox("Simple Moving Average (SMA)", True)
    sma_period = st.sidebar.slider("SMA Period", 5, 200, 20) if sma_active else 20
    
    ema_active = st.sidebar.checkbox("Exponential Moving Average (EMA)", False)
    ema_period = st.sidebar.slider("EMA Period", 5, 200, 50) if ema_active else 50
    
    bb_active = st.sidebar.checkbox("Bollinger Bands (BB)", False)
    bb_period = st.sidebar.slider("BB Period", 5, 100, 20) if bb_active else 20
    bb_std = st.sidebar.slider("BB Standard Deviation", 1.0, 4.0, 2.0, 0.5) if bb_active else 2.0
    
    st.sidebar.subheader("Subplot Indicators")
    rsi_active = st.sidebar.checkbox("Relative Strength Index (RSI)", False)
    macd_active = st.sidebar.checkbox("MACD Oscillator", False)
    atr_active = st.sidebar.checkbox("Average True Range (ATR)", False)
    
    if ticker:
        with st.spinner(f"Loading {ticker} market historicals..."):
            try:
                # Fetch data
                df = load_ticker_data(
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    interval=intervals[selected_int]
                )
                
                # Apply overlays
                overlays = []
                if sma_active:
                    df = add_sma(df, sma_period)
                    overlays.append(f"SMA_{sma_period}")
                if ema_active:
                    df = add_ema(df, ema_period)
                    overlays.append(f"EMA_{ema_period}")
                if bb_active:
                    df = add_bollinger_bands(df, bb_period, bb_std)
                    overlays.append("Bollinger Bands")
                    
                # Calculate metric values for display
                current_price = df["Close"].iloc[-1]
                prev_price = df["Close"].iloc[-2] if len(df) > 1 else current_price
                day_change = (current_price - prev_price) / prev_price
                max_price = df["High"].max()
                min_price = df["Low"].min()
                total_vol = df["Volume"].sum()
                
                # Metric Cards Bar
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Closing Price", format_currency(current_price), format_percentage(day_change))
                m_col2.metric("Interval High", format_currency(max_price))
                m_col3.metric("Interval Low", format_currency(min_price))
                m_col4.metric("Total Traded Volume", f"{total_vol:,.0f}")
                
                # Plot Candlestick
                st.subheader("Price Action Chart")
                fig = plot_candlestick_with_indicators(df, overlays, ticker)
                st.plotly_chart(fig, use_container_width=True)
                
                # Secondary indicator subplots
                if rsi_active:
                    st.subheader("Relative Strength Index (RSI)")
                    df = add_rsi(df, 14)
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI", line=dict(color="#f43f5e", width=1.5)))
                    fig_rsi.add_shape(type="line", x0=df.index[0], y0=70, x1=df.index[-1], y1=70, line=dict(color="red", dash="dash"))
                    fig_rsi.add_shape(type="line", x0=df.index[0], y0=30, x1=df.index[-1], y1=30, line=dict(color="green", dash="dash"))
                    fig_rsi.update_layout(height=200, template="plotly_dark", yaxis=dict(range=[0, 100]), margin=dict(t=20, b=20))
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                if macd_active:
                    st.subheader("MACD Oscillator")
                    df = add_macd(df)
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#3b82f6", width=1.5)))
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#f59e0b", width=1.2)))
                    fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color="rgba(147, 197, 253, 0.5)"))
                    fig_macd.update_layout(height=220, template="plotly_dark", margin=dict(t=20, b=20))
                    st.plotly_chart(fig_macd, use_container_width=True)
                    
                if atr_active:
                    st.subheader("Volatility - Average True Range (ATR)")
                    df = add_atr(df, 14)
                    fig_atr = go.Figure()
                    fig_atr.add_trace(go.Scatter(x=df.index, y=df["ATR_14"], name="ATR", line=dict(color="#a855f7", width=1.5)))
                    fig_atr.update_layout(height=180, template="plotly_dark", margin=dict(t=20, b=20))
                    st.plotly_chart(fig_atr, use_container_width=True)
                    
                # Historical Table Data Expose
                with st.expander("👁️ View Downloaded Historical Table Rows"):
                    st.dataframe(df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

# ==========================================================
# MODULE 2: STRATEGY BACKTESTER
# ==========================================================
elif app_mode == "2. Strategy Backtester":
    st.header("⚙️ Vectorized Algorithmic Backtester")
    
    # Input Selection Bar
    b_col1, b_col2, b_col3, b_col4 = st.columns([1.5, 1.5, 1, 1])
    
    with b_col1:
        backtest_ticker = st.text_input("Backtest Stock Ticker", "AAPL").strip().upper()
    with b_col2:
        strat_choice = st.selectbox("Algorithmic Trading Strategy", ["SMA Crossover", "EMA Crossover", "RSI Strategy", "MACD Signal Line"])
    with b_col3:
        b_start = st.date_input("Start Backtest", DEFAULT_START)
    with b_col4:
        b_end = st.date_input("End Backtest", DEFAULT_END)
        
    st.markdown("#### Hyperparameters & Friction")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    with p_col1:
        initial_cash = st.number_input("Starting Capital ($)", value=10000.0, step=1000.0)
    with p_col2:
        commission = st.slider("Transaction Fee / Slippage (%)", 0.0, 1.0, 0.1, 0.05) / 100.0
        
    # Show dynamic params based on strategy choice
    with p_col3:
        if strat_choice in ["SMA Crossover", "EMA Crossover"]:
            fast_period = st.number_input("Fast MA Period", value=9, min_value=2, max_value=100)
        elif strat_choice == "RSI Strategy":
            rsi_period = st.number_input("RSI Period", value=14, min_value=2, max_value=50)
        elif strat_choice == "MACD Signal Line":
            fast_period = st.number_input("MACD Fast Period", value=12, min_value=2, max_value=100)
            
    with p_col4:
        if strat_choice in ["SMA Crossover", "EMA Crossover"]:
            slow_period = st.number_input("Slow MA Period", value=21, min_value=5, max_value=300)
        elif strat_choice == "RSI Strategy":
            rsi_oversold = st.slider("RSI Buy Bound", 10.0, 50.0, 30.0)
            rsi_overbought = st.slider("RSI Sell Bound", 50.0, 90.0, 70.0)
        elif strat_choice == "MACD Signal Line":
            slow_period = st.number_input("MACD Slow Period", value=26, min_value=5, max_value=300)
            
    # Trigger Backtest
    if st.button("🚀 Run Strategy Simulation", use_container_width=True):
        with st.spinner("Executing simulation backtest..."):
            try:
                # Load Raw Ticker data
                raw_df = load_ticker_data(backtest_ticker, b_start.strftime("%Y-%m-%d"), b_end.strftime("%Y-%m-%d"))
                
                # Apply strategy logic to generate signals
                if strat_choice == "SMA Crossover":
                    signal_df = sma_crossover_strategy(raw_df, fast_period, slow_period)
                elif strat_choice == "EMA Crossover":
                    signal_df = ema_crossover_strategy(raw_df, fast_period, slow_period)
                elif strat_choice == "RSI Strategy":
                    signal_df = rsi_strategy(raw_df, rsi_period, rsi_oversold, rsi_overbought)
                elif strat_choice == "MACD Signal Line":
                    signal_df = macd_strategy(raw_df, fast_period, slow_period, 9)
                    
                # Run the backtester
                ledger, stats = run_backtest(signal_df, initial_cash, commission)
                
                # Display High-Level stats
                st.subheader("📊 Strategy vs. Buy & Hold Results")
                
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                met_col1.metric("Strategy Net Return", format_percentage(stats["total_net_return"]), delta=format_percentage(stats["total_net_return"] - stats["benchmark_total_return"]))
                met_col2.metric("Annualized CAGR", format_percentage(stats["cagr"]))
                met_col3.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
                met_col4.metric("Max Drawdown", f"{stats['max_drawdown']*100:.2f}%")
                
                # Secondary stats block
                det_col1, det_col2, det_col3, det_col4 = st.columns(4)
                det_col1.metric("Winning Trade Rate", f"{stats['win_rate']*100:.1f}%")
                det_col2.metric("Total Position Changes", f"{stats['total_trades']}")
                det_col3.metric("Friction Paid ($)", format_currency(stats["total_commissions_paid"]))
                det_col4.metric("Benchmark Total Return", format_percentage(stats["benchmark_total_return"]))
                
                # Strategy Growth Plots
                st.plotly_chart(plot_backtest_equity_curve(ledger, strat_choice), use_container_width=True)
                
                # Subplots for drawdowns and returns
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    st.plotly_chart(plot_drawdown(ledger), use_container_width=True)
                with plot_col2:
                    st.plotly_chart(plot_returns_distribution(ledger), use_container_width=True)
                    
                # Ledger inspector
                with st.expander("👁️ View Transaction Ledger & Positions Table"):
                    st.dataframe(ledger, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Strategy runtime failed: {str(e)}")

# ==========================================================
# MODULE 3: PORTFOLIO ALLOCATOR
# ==========================================================
elif app_mode == "3. Portfolio Allocator":
    st.header("💼 Multi-Asset Portfolio Management Lab")
    
    col_input, col_dates = st.columns([2, 1])
    
    with col_input:
        portfolio_input = st.text_input("Enter comma-separated assets to backtest", "AAPL, MSFT, TSLA, GLD")
    with col_dates:
        p_start = st.date_input("Start Allocation", DEFAULT_START)
        p_end = st.date_input("End Allocation", DEFAULT_END)
        
    # Split tickers
    p_tickers = [t.strip().upper() for t in portfolio_input.split(",") if t.strip()]
    
    if len(p_tickers) > 0:
        st.subheader("⚖️ Set Asset Weights Allocation")
        st.markdown("_Ensure weights sum up or distribute rationally. QuantLab auto-standardizes weights to 100% total._")
        
        sliders = {}
        # Render a sidebar or a layout of sliders
        slider_cols = st.columns(len(p_tickers))
        for idx, t in enumerate(p_tickers):
            with slider_cols[idx]:
                sliders[t] = st.slider(f"Weight {t} (%)", 0, 100, int(100/len(p_tickers)))
                
        # Run portfolio button
        if st.button("🚀 Calculate Weighted Portfolio Equity", use_container_width=True):
            with st.spinner("Compiling assets and aligning histories..."):
                try:
                    # Convert sliders to list of weights
                    total_sum = sum(sliders.values())
                    if total_sum <= 0:
                        st.error("Total weight cannot be 0. Please allocate weights.")
                    else:
                        norm_weights = [sliders[t] / total_sum for t in p_tickers]
                        
                        # Build
                        hist_df, p_stats, b_stats = build_weighted_portfolio(
                            tickers=p_tickers,
                            weights=norm_weights,
                            start_date=p_start.strftime("%Y-%m-%d"),
                            end_date=p_end.strftime("%Y-%m-%d")
                        )
                        
                        # Display Results
                        p_col1, p_col2 = st.columns([1, 2])
                        with p_col1:
                            # Pie distribution
                            st.plotly_chart(plot_portfolio_allocation(p_tickers, norm_weights), use_container_width=True)
                        with p_col2:
                            # Metric compare
                            st.markdown("### 📊 Metrics Comparison (vs Benchmark SPY)")
                            comp_df = pd.DataFrame({
                                "Metric": ["CAGR (Annualized)", "Annual Volatility", "Sharpe Ratio", "Max Drawdown", "Beta vs Benchmark", "CAPM Alpha"],
                                "Portfolio": [
                                    format_percentage(p_stats["cagr"]),
                                    format_percentage(p_stats["volatility"]),
                                    f"{p_stats['sharpe_ratio']:.2f}",
                                    f"{p_stats['max_drawdown']*100:.2f}%",
                                    f"{p_stats['beta']:.2f}",
                                    format_percentage(p_stats["alpha"])
                                ],
                                "Benchmark (SPY)": [
                                    format_percentage(b_stats["cagr"]),
                                    format_percentage(b_stats["volatility"]),
                                    f"{b_stats['sharpe_ratio']:.2f}",
                                    f"{b_stats['max_drawdown']*100:.2f}%",
                                    "1.00",
                                    "0.00%"
                                ]
                            })
                            st.table(comp_df)
                            
                        # Chart
                        st.subheader("📈 Weighted Portfolio Equity Growth ($10,000 Starting)")
                        fig_pe = go.Figure()
                        fig_pe.add_trace(go.Scatter(x=hist_df.index, y=hist_df["Portfolio_Equity"], name="My Weighted Portfolio", line=dict(color="#10b981", width=2.5)))
                        fig_pe.add_trace(go.Scatter(x=hist_df.index, y=hist_df["Benchmark_Equity"], name="SPY Benchmark", line=dict(color="#94a3b8", width=1.5, dash="dash")))
                        
                        # Also show individual stock grows
                        for t in p_tickers:
                            fig_pe.add_trace(go.Scatter(x=hist_df.index, y=hist_df[f"{t}_Growth"], name=f"{t} (Weighted Growth)", line=dict(width=1.0, opacity=0.5)))
                            
                        fig_pe.update_layout(height=450, template="plotly_dark", hovermode="x unified", legend=dict(orientation="h"))
                        st.plotly_chart(fig_pe, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Portfolio build failed: {str(e)}")

# ==========================================================
# MODULE 4: MACHINE LEARNING FEATURE EXTRACTOR
# ==========================================================
elif app_mode == "4. Machine Learning Sandbox":
    st.header("🤖 Scikit-Learn Time-Series Feature Engineering")
    st.markdown(\"\"\"
    QuantLab utilizes **Scikit-Learn** validation methods inside its pipeline. 
    This sandbox prepares raw stock data into multi-lag target-aligned matrices ready to feed standard Machine Learning models (like Linear Regression or Random Forest).
    \"\"\" )
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        ml_ticker = st.text_input("Machine Learning Target Stock", "AAPL").strip().upper()
    with m_col2:
        lag_count = st.slider("Select Feature Lag Steps", 1, 10, 5)
        
    if ml_ticker:
        with st.spinner("Generating lag datasets..."):
            try:
                # Fetch Close
                raw_df = load_ticker_data(ml_ticker, DEFAULT_START.strftime("%Y-%m-%d"), DEFAULT_END.strftime("%Y-%m-%d"))
                
                # Gen features
                X, y = create_ml_lag_features(raw_df, lag_count, "Close")
                
                st.success("Successfully generated feature arrays!")
                
                # Show matrices
                st.subheader("💡 Multi-Lag Feature Matrix (X)")
                st.markdown("_Rows aligned with historical date. Columns are preceding days' prices/returns used to make predictions._")
                st.dataframe(X.head(10), use_container_width=True)
                
                st.subheader("🎯 Shifted Target (y)")
                st.markdown("_Target Close Price for the NEXT trading day._")
                st.dataframe(pd.DataFrame({"Target_Next_Close": y}).head(10), use_container_width=True)
                
                # Training guide
                st.info(\"\"\"
                💻 **Next Research Step**:
                Run this machine learning model locally inside the python codebase using Scikit-Learn:
                \`\`\`python
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import TimeSeriesSplit
                
                # Initialize Model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Fit
                model.fit(X, y)
                \`\`\`
                \"\"\")
                
            except Exception as e:
                st.error(f"Machine learning features generation failed: {str(e)}")
`
  },
  "requirements.txt": {
    path: "requirements.txt",
    name: "requirements.txt",
    language: "text",
    content: `streamlit>=1.30.0
pandas>=2.1.0
numpy>=1.24.0
plotly>=5.18.0
yfinance>=0.2.35
scipy>=1.11.0
scikit-learn>=1.3.0
`
  },
  ".gitignore": {
    path: ".gitignore",
    name: ".gitignore",
    language: "text",
    content: `__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
*.manifest
*.spec
pip-log.txt
htmlcov/
.toml
.cache
.pytest_cache/
.coverage
db.sqlite3
docs/_build/
.ipynb_checkpoints
.streamlit/
data/*.csv
data/*.parquet
data/*.json
venv/
.env
`
  },
  "src/__init__.py": {
    path: "src/__init__.py",
    name: "__init__.py",
    language: "python",
    content: `\"\"\"
QuantLab Quantitative Library.
Provides high-performance market data retrieval, technical indicators, financial metrics, backtesting, and portfolio analytics.
\"\"\"

__version__ = "1.0.0"
__author__ = "QuantLab Core Devs"
`
  },
  "src/data_loader.py": {
    path: "src/data_loader.py",
    name: "data_loader.py",
    language: "python",
    content: `\"\"\"
Data Loader Module.
Retrieves and validates historical market data from Yahoo Finance.
\"\"\"

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
    \"\"\"
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
    \"\"\"
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
    \"\"\"
    Quick ticker verification using yfinance metadata.
    
    Args:
        ticker (str): The ticker symbol to check.
        
    Returns:
        bool: True if ticker is valid and exists, False otherwise.
    \"\"\"
    ticker_obj = yf.Ticker(ticker)
    try:
        hist = ticker_obj.history(period="1d")
        return not hist.empty
    except Exception:
        return False
`
  },
  "src/indicators.py": {
    path: "src/indicators.py",
    name: "indicators.py",
    language: "python",
    content: `\"\"\"
Technical Indicators Engine.
Provides clean and vectorized formulas for calculating standard technical analysis indicators on historical price data.
\"\"\"

import pandas as pd
import numpy as np


def add_sma(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.DataFrame:
    \"\"\"
    Appends a Simple Moving Average (SMA) column to the DataFrame.
    \"\"\"
    df = df.copy()
    df[f"SMA_{period}"] = df[column].rolling(window=period).mean()
    return df


def add_ema(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.DataFrame:
    \"\"\"
    Appends an Exponential Moving Average (EMA) column to the DataFrame.
    \"\"\"
    df = df.copy()
    df[f"EMA_{period}"] = df[column].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.DataFrame:
    \"\"\"
    Appends a Relative Strength Index (RSI) column to the DataFrame.
    Calculated using standard Wilder's smoothing technique.
    \"\"\"
    df = df.copy()
    delta = df[column].diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
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
    \"\"\"
    Appends MACD, Signal, and MACD Histogram columns to the DataFrame.
    \"\"\"
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
    \"\"\"
    Appends Bollinger Bands columns to the DataFrame.
    \"\"\"
    df = df.copy()
    df["BB_Middle"] = df[column].rolling(window=period).mean()
    rstd = df[column].rolling(window=period).std()
    
    df["BB_Upper"] = df["BB_Middle"] + (rstd * num_std)
    df["BB_Lower"] = df["BB_Middle"] - (rstd * num_std)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    \"\"\"
    Appends an Average True Range (ATR) column to the DataFrame.
    \"\"\"
    df = df.copy()
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    
    close_prev = df["Close"].shift(1).values
    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)
    
    tr = np.max(np.column_stack([tr1, tr2, tr3]), axis=1)
    
    tr_series = pd.Series(tr, index=df.index)
    atr = tr_series.rolling(window=period).mean().values
    
    for i in range(period, len(df)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
    df[f"ATR_{period}"] = atr
    return df
`
  },
  "src/metrics.py": {
    path: "src/metrics.py",
    name: "metrics.py",
    language: "python",
    content: `\"\"\"
Financial Metrics Module.
Calculates risk-adjusted returns, drawdown ratios, and regression statistics (Alpha and Beta) relative to a benchmark.
\"\"\"

from typing import Dict, Union, Optional
import numpy as np
import pandas as pd


def daily_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().fillna(0.0)


def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1)).fillna(0.0)


def cagr(series: pd.Series, periods_per_year: float = 252.0) -> float:
    if len(series) < 2 or series.iloc[0] <= 0:
        return 0.0
    total_returns = series.iloc[-1] / series.iloc[0]
    years = len(series) / periods_per_year
    if years <= 0:
        return 0.0
    return float((total_returns ** (1 / years)) - 1.0)


def volatility(returns_series: pd.Series, periods_per_year: float = 252.0) -> float:
    if len(returns_series) < 2:
        return 0.0
    return float(returns_series.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns_series: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: float = 252.0
) -> float:
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
    if len(returns_series) < 2:
        return 0.0
    ann_ret = np.mean(returns_series) * periods_per_year
    downside_returns = returns_series[returns_series < 0]
    if len(downside_returns) < 2:
        return 0.0
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    if downside_std <= 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / downside_std)


def max_drawdown(series: pd.Series) -> Dict[str, Union[float, pd.Timestamp]]:
    if len(series) < 1:
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None}
    roll_max = series.cummax()
    drawdowns = (series - roll_max) / roll_max
    max_dd = drawdowns.min()
    trough_date = drawdowns.idxmin()
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
    df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    df.columns = ["asset", "benchmark"]
    if len(df) < 5:
        return {"alpha": 0.0, "beta": 1.0}
    covariance = df["asset"].cov(df["benchmark"])
    benchmark_variance = df["benchmark"].var()
    beta = float(covariance / benchmark_variance) if benchmark_variance > 0 else 1.0
    asset_ann_ret = df["asset"].mean() * periods_per_year
    benchmark_ann_ret = df["benchmark"].mean() * periods_per_year
    alpha = float((asset_ann_ret - risk_free_rate) - beta * (benchmark_ann_ret - risk_free_rate))
    return {"alpha": alpha, "beta": beta}


def generate_full_metrics_suite(
    equity_series: pd.Series,
    benchmark_series: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, Union[float, str, None]]:
    returns = daily_returns(equity_series)
    metrics = {
        "cagr": cagr(equity_series),
        "volatility": volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(equity_series)["max_drawdown"],
        "max_drawdown_peak": str(max_drawdown(equity_series)["peak_date"].date()) if max_drawdown(equity_series)["peak_date"] else None,
        "max_drawdown_trough": str(max_drawdown(equity_series)["trough_date"].date()) if max_drawdown(equity_series)["trough_date"] else None,
        "alpha": 0.0,
        "beta": 1.0
    }
    if benchmark_series is not None:
        benchmark_rets = daily_returns(benchmark_series)
        ab = alpha_beta(returns, benchmark_rets, risk_free_rate)
        metrics["alpha"] = ab["alpha"]
        metrics["beta"] = ab["beta"]
    return metrics
`
  },
  "src/strategies.py": {
    path: "src/strategies.py",
    name: "strategies.py",
    language: "python",
    content: `\"\"\"
Trading Strategies Module.
Contains signal generation logic for technical algorithmic trading strategies.
Signals:
    1.0  = LONG / BUY
   -1.0  = SHORT / SELL
    0.0  = LIQUIDATE / FLAT / OUT
\"\"\"

import pandas as pd
from src.indicators import add_sma, add_ema, add_rsi, add_macd


def buy_and_hold_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = 1.0
    return df


def sma_crossover_strategy(
    df: pd.DataFrame,
    fast_period: int = 50,
    slow_period: int = 200
) -> pd.DataFrame:
    df = df.copy()
    df = add_sma(df, fast_period, "Close")
    df = add_sma(df, slow_period, "Close")
    fast_col = f"SMA_{fast_period}"
    slow_col = f"SMA_{slow_period}"
    df["Signal"] = 0.0
    df.loc[df[fast_col] > df[slow_col], "Signal"] = 1.0
    return df


def ema_crossover_strategy(
    df: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21
) -> pd.DataFrame:
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
            current_signal = 1.0
        elif rsi_val > overbought:
            current_signal = 0.0
        df.at[idx, "Signal"] = current_signal
    return df


def macd_strategy(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    df = df.copy()
    df = add_macd(df, fast_period, slow_period, signal_period, "Close")
    df["Signal"] = 0.0
    df.loc[df["MACD"] > df["MACD_Signal"], "Signal"] = 1.0
    return df
`
  },
  "src/portfolio.py": {
    path: "src/portfolio.py",
    name: "portfolio.py",
    language: "python",
    content: `\"\"\"
Portfolio Analytics Module.
Allows allocating weights across multiple assets, constructing aggregate portfolio equity curves, and analyzing risk/return vs a benchmark index.
\"\"\"

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
    if len(tickers) != len(weights):
        raise ValueError("Tickers and weights lists must be of equal length.")
    if not np.isclose(sum(weights), 1.0):
        weights = [w / sum(weights) for w in weights]
        
    asset_data = {}
    common_dates = []
    
    for t in tickers:
        df = load_ticker_data(t, start_date, end_date)
        asset_data[t] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        common_dates.append(df.index)
        
    try:
        benchmark_df = load_ticker_data(benchmark_ticker, start_date, end_date)
        benchmark_series = benchmark_df["Adj Close"] if "Adj Close" in benchmark_df.columns else benchmark_df["Close"]
    except Exception:
        benchmark_series = None
        
    if not common_dates:
        raise ValueError("No data could be retrieved for any specified portfolio ticker.")
        
    aligned_index = common_dates[0]
    for idx in common_dates[1:]:
        aligned_index = aligned_index.intersection(idx)
        
    if benchmark_series is not None:
        aligned_index = aligned_index.intersection(benchmark_series.index)
        
    prices_df = pd.DataFrame(index=aligned_index)
    returns_df = pd.DataFrame(index=aligned_index)
    for t in tickers:
        prices_df[t] = asset_data[t].loc[aligned_index]
        returns_df[t] = prices_df[t].pct_change().fillna(0.0)
        
    portfolio_daily_returns = returns_df.dot(weights)
    portfolio_equity = initial_capital * (1.0 + portfolio_daily_returns).cumprod()
    
    if benchmark_series is not None:
        bench_prices = benchmark_series.loc[aligned_index]
        bench_returns = bench_prices.pct_change().fillna(0.0)
        benchmark_equity = initial_capital * (1.0 + bench_returns).cumprod()
    else:
        benchmark_equity = portfolio_equity.copy()
        bench_returns = portfolio_daily_returns.copy()
        
    history_df = pd.DataFrame({
        "Portfolio_Return": portfolio_daily_returns,
        "Portfolio_Equity": portfolio_equity,
        "Benchmark_Return": bench_returns,
        "Benchmark_Equity": benchmark_equity
    }, index=aligned_index)
    
    for t in tickers:
        history_df[f"{t}_Price"] = prices_df[t]
        history_df[f"{t}_Growth"] = initial_capital * (prices_df[t] / prices_df[t].iloc[0])
        
    port_suite = generate_full_metrics_suite(portfolio_equity, benchmark_equity)
    bench_suite = generate_full_metrics_suite(benchmark_equity, benchmark_equity)
    
    return history_df, port_suite, bench_suite
`
  },
  "src/backtester.py": {
    path: "src/backtester.py",
    name: "backtester.py",
    language: "python",
    content: `\"\"\"
Backtesting Engine.
Simulates historical performance of trading signals with transaction fees and execution lag to prevent look-ahead bias.
\"\"\"

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from src.metrics import generate_full_metrics_suite


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    commission_pct: float = 0.001,
    benchmark_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if "Signal" not in df.columns:
        raise ValueError("DataFrame must contain a 'Signal' column for backtesting.")
    backtest = df.copy()
    backtest["Asset_Return"] = backtest["Close"].pct_change().fillna(0.0)
    backtest["Position"] = backtest["Signal"].shift(1).fillna(0.0)
    backtest["Strategy_Raw_Return"] = backtest["Position"] * backtest["Asset_Return"]
    backtest["Position_Change"] = backtest["Position"].diff().abs().fillna(0.0)
    if len(backtest) > 0 and backtest["Position"].iloc[0] != 0.0:
        backtest.at[backtest.index[0], "Position_Change"] = abs(backtest["Position"].iloc[0])
    backtest["Transaction_Cost"] = backtest["Position_Change"] * commission_pct
    backtest["Strategy_Net_Return"] = backtest["Strategy_Raw_Return"] - backtest["Transaction_Cost"]
    backtest["Equity_Curve"] = initial_capital * (1.0 + backtest["Strategy_Net_Return"]).cumprod()
    backtest["Benchmark_Equity_Curve"] = initial_capital * (1.0 + backtest["Asset_Return"]).cumprod()
    
    trades = backtest[backtest["Position_Change"] > 0.0]
    num_trades = len(trades)
    
    trade_profits = []
    current_trade_return = 1.0
    in_trade = False
    
    for i in range(len(backtest)):
        pos = backtest["Position"].iloc[i]
        ret = backtest["Asset_Return"].iloc[i]
        if pos != 0.0:
            in_trade = True
            current_trade_return *= (1.0 + (pos * ret))
        else:
            if in_trade:
                trade_profits.append(current_trade_return - 1.0)
                current_trade_return = 1.0
                in_trade = False
    if in_trade:
        trade_profits.append(current_trade_return - 1.0)
        
    win_rate = 0.0
    average_trade_return = 0.0
    if len(trade_profits) > 0:
        win_rate = sum(1 for r in trade_profits if r > 0.0) / len(trade_profits)
        average_trade_return = np.mean(trade_profits)
        
    benchmark_equity = backtest["Benchmark_Equity_Curve"]
    metrics = generate_full_metrics_suite(backtest["Equity_Curve"], benchmark_equity)
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
`
  },
  "src/visualizer.py": {
    path: "src/visualizer.py",
    name: "visualizer.py",
    language: "python",
    content: `\"\"\"
Plotly Visualizer Module.
Generates highly interactive financial and strategy backtesting charts for Streamlit dashboards.
\"\"\"

from typing import List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    indicators_list: Optional[List[str]] = None,
    ticker_name: str = ""
) -> go.Figure:
    indicators_list = indicators_list or []
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#10b981", decreasing_line_color="#ef4444"
    ), row=1, col=1)
    
    colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#ec4899", "#14b8a6"]
    color_idx = 0
    for col in df.columns:
        if col in indicators_list:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(width=1.5, color=colors[color_idx % len(colors)])), row=1, col=1)
            color_idx += 1
        elif col == "BB_Middle" and "Bollinger Bands" in indicators_list:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Middle"], name="BB Middle", line=dict(dash="dash", color="#3b82f6")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(color="rgba(59, 130, 246, 0.3)")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(color="rgba(59, 130, 246, 0.3)"), fill="tonexty"), row=1, col=1)
            
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume", marker_color=np.where(df["Close"] >= df["Open"], "#10b981", "#ef4444"), opacity=0.6
    ), row=2, col=1)
    
    fig.update_layout(
        title=dict(text=f"{ticker_name} - Historical Interactive Candlestick Chart", font=dict(size=18)),
        xaxis_rangeslider_visible=False, yaxis_title="Price ($)", yaxis2_title="Volume", height=650,
        hovermode="x unified", template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_backtest_equity_curve(backtest_df: pd.DataFrame, strategy_name: str = "Strategy") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Equity_Curve"], mode="lines", name=f"{strategy_name} (Net)", line=dict(color="#10b981", width=2.5)))
    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Benchmark_Equity_Curve"], mode="lines", name="Buy & Hold Benchmark", line=dict(color="#94a3b8", width=1.5, dash="dash")))
    fig.update_layout(title="Portfolio Growth Simulation ($10,000 Starting)", xaxis_title="Date", yaxis_title="Equity Value ($)", height=450, hovermode="x unified", template="plotly_dark")
    return fig


def plot_drawdown(backtest_df: pd.DataFrame) -> go.Figure:
    equity = backtest_df["Equity_Curve"]
    roll_max = equity.cummax()
    drawdowns = (equity - roll_max) / roll_max * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_df.index, y=drawdowns, mode="lines", name="Drawdown %", line=dict(color="#f87171", width=1.5), fill="tozeroy", fillcolor="rgba(248, 113, 113, 0.2)"))
    fig.update_layout(title="Historical Drawdown Profile (%)", xaxis_title="Date", yaxis_title="Drawdown (%)", yaxis=dict(ticksuffix="%"), height=300, hovermode="x unified", template="plotly_dark")
    return fig


def plot_returns_distribution(backtest_df: pd.DataFrame) -> go.Figure:
    returns = backtest_df["Strategy_Net_Return"] * 100.0
    fig = px.histogram(returns, x="Strategy_Net_Return", nbins=100, title="Daily Return Distribution Percentage", labels={"Strategy_Net_Return": "Daily Return (%)"}, color_discrete_sequence=["#3b82f6"], opacity=0.75)
    fig.update_layout(yaxis_title="Frequency (Days)", height=300, template="plotly_dark", showlegend=False)
    return fig


def plot_portfolio_allocation(tickers: List[str], weights: List[float]) -> go.Figure:
    fig = go.Figure(data=[go.Pie(labels=tickers, values=weights, hole=0.4, hoverinfo="label+percent", textinfo="label+value")])
    fig.update_layout(title="Target Asset Allocations", height=350, template="plotly_dark")
    return fig
`
  },
  "src/utils.py": {
    path: "src/utils.py",
    name: "utils.py",
    language: "python",
    content: `\"\"s
Quantitative Utility Helpers.
Provides statistical support functions, numerical cleaners, and scikit-learn lag features generators.
\"\"\"

from typing import Tuple, List, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def format_currency(val: float) -> str:
    return f"\${val:,.2f}"


def format_percentage(val: float) -> str:
    return f"{val * 100.0:+.2f}%"


def clean_outliers(series: pd.Series, stdev_limit: float = 3.0) -> pd.Series:
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
    df_feat = pd.DataFrame(index=df.index)
    lag_list = list(range(1, lags + 1)) if isinstance(lags, int) else lags
    for lag in lag_list:
        df_feat[f"{target_col}_Lag_{lag}"] = df[target_col].shift(lag)
        if include_pct_changes:
            df_feat[f"{target_col}_Lag_Pct_{lag}"] = df[target_col].pct_change(lag).shift(1)
            
    df_feat["Rolling_Volatility_5"] = df[target_col].pct_change().rolling(window=5).std().shift(1)
    df_feat["Rolling_Volatility_20"] = df[target_col].pct_change().rolling(window=20).std().shift(1)
    y = df[target_col].shift(-1)
    combined = pd.concat([df_feat, y], axis=1).dropna()
    X = combined.iloc[:, :-1]
    y_aligned = combined.iloc[:, -1]
    return X, y_aligned


def time_series_train_test_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X))
`
  }
};
