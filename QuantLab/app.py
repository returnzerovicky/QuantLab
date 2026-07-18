"""
QuantLab Streamlit Application.
The primary interactive user dashboard for historical market analysis, multi-asset portfolio management,
technical indicators, and algorithmic strategy backtesting.
"""

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
    plot_portfolio_allocation,
    plot_efficient_frontier,
    plot_correlation_heatmap,
    plot_rolling_metric
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
st.markdown("""
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
""", unsafe_allow_html=True)

# Main Banner Header
st.title("QuantLab 🔬📊")
st.markdown("---")

# Navigation Sidebar
st.sidebar.title("🎛️ Analytics Panel")
app_mode = st.sidebar.selectbox(
    "Choose Lab Module",
    [
        "1. Market Data & Indicators", 
        "2. Strategy Backtester", 
        "3. Portfolio Allocator & MPT", 
        "4. Advanced Risk & Performance", 
        "5. Machine Learning Sandbox"
    ]
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
# MODULE 3: PORTFOLIO ALLOCATOR & MPT
# ==========================================================
elif app_mode == "3. Portfolio Allocator & MPT":
    st.header("💼 Multi-Asset Portfolio Management & MPT Lab")
    st.markdown("""
    This module uses **Markowitz Modern Portfolio Theory (MPT)** to construct optimal asset allocations.
    We optimize asset weights using **Sequential Least Squares Programming (SLSQP)** to maximize the Sharpe Ratio or minimize portfolio Variance.
    """)
    
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
        st.markdown("_QuantLab automatically standardizes weights to sum up to 100%._")
        
        # Save to session state so Risk Dashboard can access it
        st.session_state.portfolio_tickers = p_tickers
        st.session_state.p_start = p_start
        st.session_state.p_end = p_end
        
        sliders = {}
        slider_cols = st.columns(len(p_tickers))
        for idx, t in enumerate(p_tickers):
            with slider_cols[idx]:
                sliders[t] = st.slider(f"Weight {t} (%)", 0, 100, int(100/len(p_tickers)))
                
        # Run portfolio button
        if st.button("🚀 Calculate Weighted Portfolio & MPT Frontier", use_container_width=True):
            with st.spinner("Compiling assets, aligning histories and optimizing..."):
                try:
                    # Convert sliders to list of weights
                    total_sum = sum(sliders.values())
                    if total_sum <= 0:
                        st.error("Total weight cannot be 0. Please allocate weights.")
                    else:
                        norm_weights = [sliders[t] / total_sum for t in p_tickers]
                        st.session_state.portfolio_weights = norm_weights
                        
                        # Build
                        hist_df, p_stats, b_stats, mpt_results = build_weighted_portfolio(
                            tickers=p_tickers,
                            weights=norm_weights,
                            start_date=p_start.strftime("%Y-%m-%d"),
                            end_date=p_end.strftime("%Y-%m-%d")
                        )
                        
                        # Cache in session state
                        st.session_state.hist_df = hist_df
                        st.session_state.p_stats = p_stats
                        st.session_state.b_stats = b_stats
                        st.session_state.mpt_results = mpt_results
                        
                        # Display Results
                        p_col1, p_col2 = st.columns([1, 2])
                        with p_col1:
                            st.plotly_chart(plot_portfolio_allocation(p_tickers, norm_weights), use_container_width=True)
                        with p_col2:
                            st.markdown("### 📊 Portfolio Metrics (vs Benchmark SPY)")
                            comp_df = pd.DataFrame({
                                "Metric": [
                                    "CAGR (Annualized Return)", 
                                    "Annual Volatility", 
                                    "Sharpe Ratio (Rf=2%)", 
                                    "Max Drawdown", 
                                    "Beta vs Benchmark", 
                                    "CAPM Alpha",
                                    "Diversification Score"
                                ],
                                "Your Portfolio": [
                                    format_percentage(p_stats["cagr"]),
                                    format_percentage(p_stats["volatility"]),
                                    f"{p_stats['sharpe_ratio']:.2f}",
                                    f"{p_stats['max_drawdown']*100:.2f}%",
                                    f"{p_stats['beta']:.2f}",
                                    format_percentage(p_stats["alpha"]),
                                    f"{mpt_results['diversification_ratio']:.2f}"
                                ],
                                "Benchmark (SPY)": [
                                    format_percentage(b_stats["cagr"]),
                                    format_percentage(b_stats["volatility"]),
                                    f"{b_stats['sharpe_ratio']:.2f}",
                                    f"{b_stats['max_drawdown']*100:.2f}%",
                                    "1.00",
                                    "0.00%",
                                    "1.00"
                                ]
                            })
                            st.table(comp_df)
                            
                        # Chart
                        st.subheader("📈 Weighted Portfolio Equity Growth ($10,000 Starting)")
                        fig_pe = go.Figure()
                        fig_pe.add_trace(go.Scatter(x=hist_df.index, y=hist_df["Portfolio_Equity"], name="Your Portfolio", line=dict(color="#10b981", width=2.5)))
                        fig_pe.add_trace(go.Scatter(x=hist_df.index, y=hist_df["Benchmark_Equity"], name="SPY Benchmark", line=dict(color="#94a3b8", width=1.5, dash="dash")))
                        
                        # Also show individual stock grows
                        for t in p_tickers:
                            fig_pe.add_trace(go.Scatter(x=hist_df.index, y=hist_df[f"{t}_Growth"], name=f"{t} (Weighted Growth)", line=dict(width=1.0, opacity=0.5)))
                            
                        fig_pe.update_layout(height=450, template="plotly_dark", hovermode="x unified", legend=dict(orientation="h"))
                        st.plotly_chart(fig_pe, use_container_width=True)
                        
                        # MPT Analytics
                        st.markdown("---")
                        st.subheader("🔬 Mean-Variance Optimization & Efficient Frontier")
                        
                        col_mpt1, col_mpt2 = st.columns([1, 1])
                        with col_mpt1:
                            st.plotly_chart(plot_efficient_frontier(mpt_results, p_stats["cagr"], p_stats["volatility"]), use_container_width=True)
                        with col_mpt2:
                            st.plotly_chart(plot_correlation_heatmap(mpt_results["correlation"]), use_container_width=True)
                            
                        st.markdown("### 🏛️ MPT Optimal Portfolio Weights")
                        opt_df = pd.DataFrame({
                            "Asset": p_tickers,
                            "Your Weight (%)": [w * 100 for w in norm_weights],
                            "Max Sharpe Weight (%)": [mpt_results["max_sharpe"]["weights"].get(t, 0) * 100 for t in p_tickers],
                            "Min Variance Weight (%)": [mpt_results["min_variance"]["weights"].get(t, 0) * 100 for t in p_tickers]
                        })
                        st.dataframe(opt_df.style.format({
                            "Your Weight (%)": "{:.2f}%",
                            "Max Sharpe Weight (%)": "{:.2f}%",
                            "Min Variance Weight (%)": "{:.2f}%"
                        }), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Portfolio build failed: {str(e)}")

# ==========================================================
# MODULE 4: ADVANCED RISK & PERFORMANCE DASHBOARD
# ==========================================================
elif app_mode == "4. Advanced Risk & Performance":
    st.header("🛡️ Institutional Risk & Performance Analytics")
    st.markdown("""
    This section evaluates high-fidelity risk metrics like **Value at Risk (VaR)**, **Conditional Value at Risk (CVaR)**, 
    and annualized performance attribution metrics.
    """)
    
    # Check if a portfolio has been run in session state, if not initialize it
    if "hist_df" not in st.session_state:
        st.info("💡 Run the 'Portfolio Allocator & MPT' first to load customized asset allocations, or run with default AAPL/MSFT/TSLA/GLD below.")
        if st.button("🚀 Load Default Portfolio Analytics"):
            with st.spinner("Loading defaults..."):
                try:
                    p_tickers = ["AAPL", "MSFT", "TSLA", "GLD"]
                    norm_weights = [0.25, 0.25, 0.25, 0.25]
                    st.session_state.portfolio_tickers = p_tickers
                    st.session_state.portfolio_weights = norm_weights
                    st.session_state.p_start = DEFAULT_START
                    st.session_state.p_end = DEFAULT_END
                    
                    hist_df, p_stats, b_stats, mpt_results = build_weighted_portfolio(
                        tickers=p_tickers,
                        weights=norm_weights,
                        start_date=DEFAULT_START.strftime("%Y-%m-%d"),
                        end_date=DEFAULT_END.strftime("%Y-%m-%d")
                    )
                    st.session_state.hist_df = hist_df
                    st.session_state.p_stats = p_stats
                    st.session_state.b_stats = b_stats
                    st.session_state.mpt_results = mpt_results
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading defaults: {str(e)}")
    
    if "hist_df" in st.session_state:
        hist_df = st.session_state.hist_df
        p_stats = st.session_state.p_stats
        b_stats = st.session_state.b_stats
        mpt_results = st.session_state.mpt_results
        p_tickers = st.session_state.portfolio_tickers
        
        # Risk Metric Blocks
        st.subheader("⚠️ Tail Risk & Loss Forecasting")
        st.markdown("_Value at Risk (VaR) measures the maximum expected loss at a given confidence interval over a 1-day horizon._")
        
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        r_col1.metric("Historical VaR (95%)", f"-{p_stats['var_historical']*100:.2f}%", help="Maximum 1-day expected loss at 95% confidence interval using historical simulation.")
        r_col2.metric("Parametric VaR (95%)", f"-{p_stats['var_parametric']*100:.2f}%", help="Maximum 1-day expected loss at 95% confidence interval using parametric variance-covariance assumption.")
        r_col3.metric("Conditional VaR / ES (95%)", f"-{p_stats['cvar']*100:.2f}%", help="Expected Shortfall (average loss in the worst 5% of trading days).")
        r_col4.metric("Profit Factor", f"{p_stats['profit_factor']:.2f}", help="Sum of positive daily returns / Absolute sum of negative daily returns.")
        
        # Performance Attribution Table
        st.subheader("📊 Performance Attribution & Ratio Analysis")
        perf_df = pd.DataFrame({
            "Performance Metric": [
                "Compound Annual Growth (CAGR)",
                "Annualized Volatility",
                "Sharpe Ratio (Rf=2%)",
                "Sortino Ratio (Downside Deviation)",
                "Max Peak-to-Trough Drawdown",
                "Calmar Ratio (CAGR / Max DD)",
                "Treynor Ratio (CAGR / Beta)",
                "Information Ratio (Active return / Tracking error)",
                "Systematic Beta vs S&P500",
                "CAPM Annualized Alpha"
            ],
            "Portfolio": [
                format_percentage(p_stats["cagr"]),
                format_percentage(p_stats["volatility"]),
                f"{p_stats['sharpe_ratio']:.2f}",
                f"{p_stats['sortino_ratio']:.2f}",
                f"{p_stats['max_drawdown']*100:.2f}%",
                f"{p_stats['calmar_ratio']:.2f}",
                f"{p_stats['treynor_ratio']:.2f}",
                f"{p_stats['information_ratio']:.2f}",
                f"{p_stats['beta']:.2f}",
                format_percentage(p_stats["alpha"])
            ],
            "Benchmark (SPY)": [
                format_percentage(b_stats["cagr"]),
                format_percentage(b_stats["volatility"]),
                f"{b_stats['sharpe_ratio']:.2f}",
                f"{b_stats['sortino_ratio']:.2f}",
                f"{b_stats['max_drawdown']*100:.2f}%",
                f"{b_stats['calmar_ratio']:.2f}",
                "N/A",
                "0.00",
                "1.00",
                "0.00%"
            ]
        })
        st.table(perf_df)
        
        # Rolling Risk Plots
        st.subheader("🔄 Rolling Risk Analytics")
        rolling_window = st.slider("Rolling Volatility/Sharpe Window (Days)", 10, 126, 20)
        
        # Calculate rolling metrics
        from src.metrics import rolling_volatility, rolling_sharpe_ratio
        r_vol = rolling_volatility(hist_df["Portfolio_Return"], rolling_window)
        r_sharpe = rolling_sharpe_ratio(hist_df["Portfolio_Return"], rolling_window)
        
        col_roll1, col_roll2 = st.columns(2)
        with col_roll1:
            st.plotly_chart(plot_rolling_metric(r_vol, f"Rolling {rolling_window}-Day Volatility (Annualized)", "%", "#f43f5e"), use_container_width=True)
        with col_roll2:
            st.plotly_chart(plot_rolling_metric(r_sharpe, f"Rolling {rolling_window}-Day Sharpe Ratio (Annualized)", "", "#10b981"), use_container_width=True)
            
        # Export features
        st.markdown("---")
        st.subheader("📥 Export Performance Reports & Analytics Data")
        
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            csv_data = hist_df.to_csv()
            st.download_button(
                label="📥 Export Historical Portfolio & Asset Series (CSV)",
                data=csv_data,
                file_name="quantlab_portfolio_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_ex2:
            tickers_str = ", ".join(p_tickers)
            weights_str = ", ".join([f"{t}: {w*100:.1f}%" for t, w in zip(p_tickers, st.session_state.portfolio_weights)])
            
            report_html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 30px; color: #333; }}
                    h1 {{ color: #1e3a8a; border-bottom: 2px solid #1e3a8a; padding-bottom: 10px; }}
                    h2 {{ color: #2563eb; margin-top: 25px; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                    th {{ background-color: #f3f4f6; color: #1e3a8a; }}
                    .highlight {{ font-weight: bold; color: #10b981; }}
                    .footer {{ margin-top: 50px; font-size: 11px; color: #777; border-top: 1px solid #ddd; padding-top: 10px; }}
                </style>
            </head>
            <body>
                <h1>QuantLab Proprietary Firm Performance Report</h1>
                <p><strong>Generated Date:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Assets Analyzed:</strong> {tickers_str}</p>
                <p><strong>Allocated Portfolio Weights:</strong> {weights_str}</p>
                
                <h2>1. Advanced Risk & Performance Attribution</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Performance Metric</th>
                            <th>Portfolio</th>
                            <th>S&P 500 (SPY) Benchmark</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Compound Annual Growth (CAGR)</td><td>{format_percentage(p_stats["cagr"])}</td><td>{format_percentage(b_stats["cagr"])}</td></tr>
                        <tr><td>Annualized Volatility</td><td>{format_percentage(p_stats["volatility"])}</td><td>{format_percentage(b_stats["volatility"])}</td></tr>
                        <tr><td>Sharpe Ratio (Rf=2%)</td><td>{p_stats['sharpe_ratio']:.2f}</td><td>{b_stats['sharpe_ratio']:.2f}</td></tr>
                        <tr><td>Sortino Ratio</td><td>{p_stats['sortino_ratio']:.2f}</td><td>{b_stats['sortino_ratio']:.2f}</td></tr>
                        <tr><td>Maximum Drawdown</td><td>{p_stats['max_drawdown']*100:.2f}%</td><td>{b_stats['max_drawdown']*100:.2f}%</td></tr>
                        <tr><td>Calmar Ratio</td><td>{p_stats['calmar_ratio']:.2f}</td><td>{b_stats['calmar_ratio']:.2f}</td></tr>
                        <tr><td>Information Ratio</td><td>{p_stats['information_ratio']:.2f}</td><td>0.00</td></tr>
                        <tr><td>Treynor Ratio</td><td>{p_stats['treynor_ratio']:.2f}</td><td>N/A</td></tr>
                        <tr><td>Systematic Beta</td><td>{p_stats['beta']:.2f}</td><td>1.00</td></tr>
                        <tr><td>Annualized Alpha</td><td>{format_percentage(p_stats["alpha"])}</td><td>0.00%</td></tr>
                    </tbody>
                </table>
                
                <h2>2. Value at Risk (VaR) & Tail Loss Limits</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Tail Risk Metric</th>
                            <th>Value</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Historical VaR (95%)</td><td>-{p_stats['var_historical']*100:.2f}%</td><td>Maximum expected daily loss at 95% confidence level.</td></tr>
                        <tr><td>Parametric VaR (95%)</td><td>-{p_stats['var_parametric']*100:.2f}%</td><td>Standard Normal expected daily loss at 95% confidence.</td></tr>
                        <tr><td>Conditional VaR / Expected Shortfall</td><td>-{p_stats['cvar']*100:.2f}%</td><td>Average loss expected in the worst 5% tail events.</td></tr>
                        <tr><td>Daily Profit Factor</td><td>{p_stats['profit_factor']:.2f}</td><td>Ratio of gross gains over gross losses.</td></tr>
                        <tr><td>Portfolio Diversification Score (DR)</td><td>{mpt_results['diversification_ratio']:.2f}</td><td>A higher score (>1.0) denotes positive covariance reduction benefits.</td></tr>
                    </tbody>
                </table>
                
                <div class="footer">
                    CONFIDENTIAL | GENERATED SECURELY BY QUANTLAB INSIDE INSTITUTIONAL DEVELOPMENT WORKSPACE.
                </div>
            </body>
            </html>
            """
            st.download_button(
                label="📄 Export Printable PDF-ready Performance Report (HTML format)",
                data=report_html,
                file_name="quantlab_institutional_report.html",
                mime="text/html",
                use_container_width=True
            )

# ==========================================================
# MODULE 5: MACHINE LEARNING FEATURE EXTRACTOR
# ==========================================================
elif app_mode == "5. Machine Learning Sandbox":
    st.header("🤖 Scikit-Learn Time-Series Feature Engineering")
    st.markdown("""
    QuantLab utilizes **Scikit-Learn** validation methods inside its pipeline. 
    This sandbox prepares raw stock data into multi-lag target-aligned matrices ready to feed standard Machine Learning models (like Linear Regression or Random Forest).
    """)
    
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
                st.info("""
                💻 **Next Research Step**:
                Run this machine learning model locally inside the python codebase using Scikit-Learn:
                ```python
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import TimeSeriesSplit
                
                # Initialize Model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Fit
                model.fit(X, y)
                ```
                """)
                
            except Exception as e:
                st.error(f"Machine learning features generation failed: {str(e)}")
