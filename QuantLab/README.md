# QuantLab 🔬📊
> **Institutional-Grade Quantitative Research, Modern Portfolio Theory (MPT), & Backtesting Workspace**

QuantLab is a modular, high-performance quantitative research and asset allocation ecosystem built to simulate, optimize, and analyze proprietary trading strategies and multi-asset portfolios. 

Designed for investment analysts, proprietary traders, and portfolio managers, QuantLab leverages vectorized computation engines to run high-fidelity backtests, trace the Markowitz Efficient Frontier, and calculate institutional-grade risk attribution statistics (VaR, CVaR, Alpha, Beta, Treynor, Calmar, and Information Ratios).

---

## 🚀 Key Modules & Capabilities

### 1. Market Data & Technical Indicators (`src/indicators.py`)
*   **Vectorized Calculations**: Robust, memory-efficient Pandas and NumPy calculations on historical price series.
*   **Indicator Coverage**:
    *   **Simple Moving Average (SMA)** & **Exponential Moving Average (EMA)**
    *   **Relative Strength Index (RSI)** with standard Wilder's smoothing.
    *   **MACD (Moving Average Convergence Divergence)** with signal line crossing thresholds.
    *   **Bollinger Bands** with dynamic volatility bands.
    *   **Average True Range (ATR)** for institutional volatility positioning.

### 2. Modern Portfolio Theory & Weight Optimization (`src/portfolio.py`)
*   **Mean-Variance Optimization**: Utilizes **Sequential Least Squares Programming (SLSQP)** via Scipy to find optimal asset weights:
    *   **Maximum Sharpe Portfolio**: Optimizes weights to maximize risk-adjusted return relative to volatility.
    *   **Minimum Variance Portfolio**: Minimizes total portfolio variance to protect tail risk.
*   **Efficient Frontier Tracing**: Generates the complete hyperbola of optimal portfolios for various target return thresholds.
*   **Diversification Score**: Tracks portfolio-wide diversification ratios to ensure independent systemic covariance.
*   **Correlation & Covariance Maps**: Computes and charts underlying asset-class correlations.

### 3. Institutional Risk & Attribution (`src/metrics.py`)
*   **Tail Risk Estimation**:
    *   **Historical Value at Risk (VaR)**: Non-parametric 1-day value-at-risk at 95% confidence level.
    *   **Parametric VaR**: Parametric variance-covariance normal-distribution value-at-risk.
    *   **Expected Shortfall (Conditional VaR / CVaR)**: Tracks the average loss in the worst 5% tail events.
*   **Performance Metrics**:
    *   **CAGR**: Compound Annual Growth Rate.
    *   **Sharpe Ratio**: Annualized return per unit of total risk (risk-free rate = 2%).
    *   **Sortino Ratio**: Adjusted return per unit of downside deviation.
    *   **Calmar Ratio**: Annualized return relative to maximum peak-to-trough drawdown.
    *   **Treynor Ratio**: Excess annualized return per unit of systematic risk (Beta).
    *   **Information Ratio**: Active portfolio return over tracking error against benchmark index (SPY).
    *   **Jensen's Alpha & Systematic Beta**: CAPM regression parameters relative to S&P 500.

### 4. Vectorized Strategy Backtesting (`src/backtester.py`)
*   **Execution Lag Protection**: Enforces a strict 1-day execution delay between signal generation and execution to prevent look-ahead bias.
*   **Transaction Costs**: Models proportional commission fees (default 10 bps) on position entries and exits.
*   **Historical Drawdown Profile**: Computes continuous peak-to-trough series and tracks timing coordinates of maximum drawdowns.

---

## 🗄️ Folder Structure

```text
QuantLab/
│
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions Automated CI/CD pipeline
│
├── src/
│   ├── __init__.py         # Package entry point
│   ├── data_loader.py      # Yahoo Finance API data interface & local CSV caching
│   ├── indicators.py       # Technical indicators calculation engine
│   ├── metrics.py          # Advanced risk, return, and CAPM attribution analytics
│   ├── strategies.py       # Core strategy rule definition modules
│   ├── portfolio.py        # MPT optimization algorithms & SLSQP solvers
│   ├── backtester.py       # Execution-lagged trading simulation engine
│   ├── visualizer.py       # Interactive Plotly visualizers (heatmaps, frontiers)
│   └── utils.py            # Quantitative math formatting & logging utilities
│
├── tests/
│   ├── test_metrics.py     # Unit tests verifying financial calculations
│   └── test_portfolio.py   # Unit tests verifying SLSQP solvers & covariance logic
│
├── app.py                  # Streamlit Multi-Module Lab Dashboard (UI)
├── config.json             # Global static configuration parameters
├── requirements.txt        # PIP environment dependencies
├── Dockerfile              # Docker container deployment blueprint
├── .gitignore              # Ignored compilation files & cache paths
└── README.md               # Repository documentation
```

---

## 🔬 Mathematical Formulas & Models

### 1. Modern Portfolio Theory (Mean-Variance Optimization)
For a portfolio with $N$ assets, weights $w$, expected returns $\mu$, and covariance matrix $\Sigma$:

*   **Portfolio Return**: 
    $$R_p = w^T \mu$$
*   **Portfolio Volatility**: 
    $$\sigma_p = \sqrt{w^T \Sigma w}$$
*   **Maximum Sharpe Portfolio Optimization Target**:
    $$\max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}} \quad \text{subject to} \quad \sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le 1$$
*   **Minimum Variance Portfolio Optimization Target**:
    $$\min_w w^T \Sigma w \quad \text{subject to} \quad \sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le 1$$

---

### 2. Performance & Risk Attribution
*   **Jensen's Alpha & CAPM Systematic Beta**:
    $$R_{p,t} - R_f = \alpha + \beta (R_{m,t} - R_f) + \epsilon_t$$
*   **Downside Deviation (for Sortino Ratio)**:
    $$\sigma_{down} = \sqrt{\frac{1}{T}\sum_{t=1}^T \min(R_t - R_{target}, 0)^2}$$
*   **Treynor Ratio**:
    $$\text{Treynor Ratio} = \frac{R_p - R_f}{\beta_p}$$
*   **Information Ratio**:
    $$\text{Information Ratio} = \frac{R_p - R_m}{\sigma(R_p - R_m)}$$

---

### 3. Tail Risk Management (Value at Risk)
*   **Historical VaR (95%)**:
    $$\text{VaR}_{0.95} = - \text{Percentile}(R, 5\%)$$
*   **Parametric VaR (95%)**:
    $$\text{VaR}_{0.95} = - (\mu_R + z_{0.05} \cdot \sigma_R)$$
*   **Expected Shortfall (Conditional VaR - CVaR)**:
    $$\text{CVaR}_{0.95} = E[-R \mid -R \ge \text{VaR}_{0.95}]$$

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.10, 3.11 or 3.12
*   Docker (Optional for containerization)

### Local Environment Setup
1.  **Clone the workspace**:
    ```bash
    git clone https://github.com/yourusername/QuantLab.git
    cd QuantLab
    ```
2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```
3.  **Install the dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## 📊 How to Run the Ecosystem

### 1. Launching the Interactive UI Dashboard
```bash
streamlit run app.py
```
This boots the Streamlit server on your machine (usually http://localhost:8501).

### 2. Running Unit Tests
Validate all financial math and SLSQP optimization logic locally using:
```bash
PYTHONPATH=. python3 tests/test_metrics.py
PYTHONPATH=. python3 tests/test_portfolio.py
```

### 3. Containerized Deployment (Docker)
1.  **Build the Docker Image**:
    ```bash
    docker build -t quantlab-app .
    ```
2.  **Run the Container**:
    ```bash
    docker run -d -p 8501:8501 --name quantlab-running quantlab-app
    ```
The containerized dashboard will be instantly active at `http://localhost:8501`.

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

