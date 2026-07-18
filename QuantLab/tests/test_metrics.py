import unittest
import numpy as np
import pandas as pd
from src.metrics import (
    cagr,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    historical_var,
    parametric_var,
    conditional_var_es,
    profit_factor
)

class TestFinancialMetrics(unittest.TestCase):
    def setUp(self):
        # Create deterministic price series
        # Initial: 100, then growing 1% per day for 252 days
        self.dates = pd.date_range(start="2024-01-01", periods=252, freq="D")
        self.prices = pd.Series([100.0 * (1.01 ** i) for i in range(252)], index=self.dates)
        self.returns = self.prices.pct_change().fillna(0.0)

    def test_cagr_positive_growth(self):
        calculated_cagr = cagr(self.prices, periods_per_year=252)
        # 1.01 ** 251 should translate to high CAGR. Let's verify it is strictly positive.
        self.assertTrue(calculated_cagr > 0.0)
        self.assertAlmostEqual(calculated_cagr, (1.01 ** 252) - 1.0, places=4)

    def test_volatility_zero_variance(self):
        # Constant returns of exactly 1%
        const_returns = pd.Series([0.01] * 100)
        calculated_vol = volatility(const_returns, periods_per_year=252)
        self.assertAlmostEqual(calculated_vol, 0.0, places=6)

    def test_sharpe_ratio(self):
        # High returns, low risk-free-rate should give high Sharpe Ratio
        high_returns = pd.Series([0.02, 0.02, 0.02])
        # Sharpe ratio with 0 volatility will be handled gracefully
        ratio = sharpe_ratio(high_returns, risk_free_rate=0.01)
        self.assertEqual(ratio, 0.0)

    def test_max_drawdown(self):
        # Create a peak-trough price series
        # Peak at index 1 (110), trough at index 3 (90), recovering to 100
        declining_prices = pd.Series([100, 110, 95, 90, 100], index=pd.date_range("2024-01-01", periods=5))
        dd_res = max_drawdown(declining_prices)
        # Expected max drawdown = (90 - 110) / 110 = -0.181818
        self.assertAlmostEqual(dd_res["max_drawdown"], -0.181818, places=4)
        self.assertEqual(dd_res["peak_date"], declining_prices.index[1])
        self.assertEqual(dd_res["trough_date"], declining_prices.index[3])

    def test_profit_factor(self):
        # 3 positive returns of +1%, 1 negative return of -1%
        rets = pd.Series([0.01, 0.01, -0.01, 0.01])
        pf = profit_factor(rets)
        # Gross profits = 0.03, Gross losses = 0.01. PF = 3.0
        self.assertAlmostEqual(pf, 3.0, places=6)

    def test_value_at_risk(self):
        # Standard normal distribution returns
        np.random.seed(42)
        normal_rets = pd.Series(np.random.normal(0.001, 0.01, 1000))
        hist_v = historical_var(normal_rets, confidence=0.95)
        param_v = parametric_var(normal_rets, confidence=0.95)
        cvar_v = conditional_var_es(normal_rets, confidence=0.95)
        
        # VaR should be positive loss estimates (defined as positive quantities)
        self.assertTrue(hist_v > 0)
        self.assertTrue(param_v > 0)
        self.assertTrue(cvar_v > hist_v)  # CVaR is always worse than standard VaR

if __name__ == "__main__":
    unittest.main()
