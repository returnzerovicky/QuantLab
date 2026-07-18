"""
Plotly Visualizer Module.
Generates highly interactive financial and strategy backtesting charts for Streamlit dashboards.
"""

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
    """
    Creates an interactive candlestick chart overlayed with technical indicators and a volume subplot.
    """
    indicators_list = indicators_list or []
    
    # Create subplots: Row 1 is Price, Row 2 is Volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#10b981", # Emerald
            decreasing_line_color="#ef4444"  # Red
        ),
        row=1,
        col=1
    )
    
    # Overlay indicators
    colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#ec4899", "#14b8a6"] # blue, purple, amber, pink, teal
    color_idx = 0
    
    for col in df.columns:
        if col in indicators_list:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1.5, color=colors[color_idx % len(colors)])
                ),
                row=1,
                col=1
            )
            color_idx += 1
            
        # Draw Bollinger Bands specifically if requested
        elif col == "BB_Middle" and "Bollinger Bands" in indicators_list:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Middle"], name="BB Middle", line=dict(dash="dash", color="#3b82f6")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(color="rgba(59, 130, 246, 0.3)")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(color="rgba(59, 130, 246, 0.3)"), fill="tonexty"), row=1, col=1)
            
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=np.where(df["Close"] >= df["Open"], "#10b981", "#ef4444"),
            opacity=0.6
        ),
        row=2,
        col=1
    )
    
    # Layout styles
    fig.update_layout(
        title=dict(text=f"{ticker_name} - Historical Interactive Candlestick Chart", font=dict(size=18)),
        xaxis_rangeslider_visible=False,
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        height=650,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_backtest_equity_curve(
    backtest_df: pd.DataFrame,
    strategy_name: str = "Strategy"
) -> go.Figure:
    """
    Plots the comparative growth of $10,000 for the backtested strategy vs a Buy & Hold benchmark.
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=backtest_df["Equity_Curve"],
            mode="lines",
            name=f"{strategy_name} (Net)",
            line=dict(color="#10b981", width=2.5)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=backtest_df["Benchmark_Equity_Curve"],
            mode="lines",
            name="Buy & Hold Benchmark",
            line=dict(color="#94a3b8", width=1.5, dash="dash")
        )
    )
    
    # Color shading where strategy outperforms benchmark
    fig.update_layout(
        title="Portfolio Growth Simulation ($10,000 Starting)",
        xaxis_title="Date",
        yaxis_title="Equity Value ($)",
        height=450,
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    
    return fig


def plot_drawdown(backtest_df: pd.DataFrame) -> go.Figure:
    """
    Plots the daily drawdown percentage curve (shaded red).
    """
    equity = backtest_df["Equity_Curve"]
    roll_max = equity.cummax()
    drawdowns = (equity - roll_max) / roll_max * 100.0
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=drawdowns,
            mode="lines",
            name="Drawdown %",
            line=dict(color="#f87171", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(248, 113, 113, 0.2)"
        )
    )
    
    fig.update_layout(
        title="Historical Drawdown Profile (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis=dict(ticksuffix="%"),
        height=300,
        hovermode="x unified",
        template="plotly_dark"
    )
    
    return fig


def plot_returns_distribution(backtest_df: pd.DataFrame) -> go.Figure:
    """
    Generates a daily returns histogram with an overlaid normal distribution line.
    """
    returns = backtest_df["Strategy_Net_Return"] * 100.0 # scale to percent
    
    fig = px.histogram(
        returns,
        x="Strategy_Net_Return",
        nbins=100,
        title="Daily Return Distribution Percentage",
        labels={"Strategy_Net_Return": "Daily Return (%)"},
        color_discrete_sequence=["#3b82f6"],
        opacity=0.75
    )
    
    fig.update_layout(
        yaxis_title="Frequency (Days)",
        height=300,
        template="plotly_dark",
        showlegend=False
    )
    
    return fig


def plot_portfolio_allocation(tickers: List[str], weights: List[float]) -> go.Figure:
    """
    Plots a beautiful donut pie chart of portfolio asset allocations.
    """
    fig = go.Figure(
        data=[
            go.Pie(
                labels=tickers,
                values=weights,
                hole=0.4,
                hoverinfo="label+percent",
                textinfo="label+value",
                marker=dict(colors=px.colors.qualitative.Dark24)
            )
        ]
    )
    
    fig.update_layout(
        title="Target Asset Allocations",
        height=350,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def plot_efficient_frontier(
    mpt_results: dict,
    current_ret: Optional[float] = None,
    current_vol: Optional[float] = None
) -> go.Figure:
    """
    Plots the Efficient Frontier curve with highlights on Maximum Sharpe,
    Minimum Variance, and Current/Allocated portfolios.
    """
    fig = go.Figure()
    
    # Extract frontier points
    frontier = mpt_results.get("frontier", [])
    if frontier:
        vols = [pt["volatility"] * 100.0 for pt in frontier]
        rets = [pt["return"] * 100.0 for pt in frontier]
        fig.add_trace(go.Scatter(
            x=vols,
            y=rets,
            mode="lines+markers",
            name="Efficient Frontier",
            line=dict(color="#10b981", width=3, shape="spline"),
            marker=dict(size=5, color="#34d399")
        ))
        
    # Add Max Sharpe portfolio
    max_s = mpt_results.get("max_sharpe", {})
    if max_s:
        fig.add_trace(go.Scatter(
            x=[max_s["volatility"] * 100.0],
            y=[max_s["return"] * 100.0],
            mode="markers+text",
            name="Max Sharpe Portfolio",
            text=["★ Max Sharpe"],
            textposition="top center",
            marker=dict(color="#fbbf24", size=15, symbol="star")
        ))
        
    # Add Min Variance portfolio
    min_v = mpt_results.get("min_variance", {})
    if min_v:
        fig.add_trace(go.Scatter(
            x=[min_v["volatility"] * 100.0],
            y=[min_v["return"] * 100.0],
            mode="markers+text",
            name="Min Variance Portfolio",
            text=["✦ Min Var"],
            textposition="bottom center",
            marker=dict(color="#ef4444", size=15, symbol="diamond")
        ))
        
    # Add Current allocated portfolio
    if current_ret is not None and current_vol is not None:
        fig.add_trace(go.Scatter(
            x=[current_vol * 100.0],
            y=[current_ret * 100.0],
            mode="markers+text",
            name="Your Allocation",
            text=["● Your Allocation"],
            textposition="middle right",
            marker=dict(color="#3b82f6", size=14, symbol="circle")
        ))
        
    fig.update_layout(
        title="Modern Portfolio Theory: Efficient Frontier",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Expected Annualized Return (%)",
        height=400,
        hovermode="closest",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig


def plot_correlation_heatmap(corr_matrix_dict: dict) -> go.Figure:
    """
    Generates an interactive Correlation Heatmap.
    """
    df_corr = pd.DataFrame(corr_matrix_dict)
    tickers = df_corr.columns.tolist()
    
    fig = go.Figure(data=go.Heatmap(
        z=df_corr.values,
        x=tickers,
        y=tickers,
        colorscale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Correlation"),
        hovertemplate="Asset A: %{x}<br>Asset B: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
    ))
    
    # Annotate values
    for i, row in enumerate(df_corr.values):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=tickers[j],
                y=tickers[i],
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(val) > 0.5 else "black", size=11)
            )
            
    fig.update_layout(
        title="Asset Correlation Matrix",
        height=400,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_rolling_metric(
    rolling_series: pd.Series,
    title: str,
    yaxis_suffix: str = "",
    color: str = "#3b82f6"
) -> go.Figure:
    """
    Plots a single line chart for rolling metrics (e.g. Rolling Volatility or Rolling Sharpe).
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_series.index,
        y=rolling_series,
        mode="lines",
        name=title,
        line=dict(color=color, width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=title,
        yaxis=dict(ticksuffix=yaxis_suffix),
        height=300,
        hovermode="x unified",
        template="plotly_dark"
    )
    return fig

