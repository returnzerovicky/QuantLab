import unittest
import numpy as np
import pandas as pd
from src.portfolio import (
    optimize_portfolio_mpt,
    calculate_diversification_ratio
)

class TestPortfolioOptimization(unittest.TestCase):
    def setUp(self):
        # Setup simulated price dataframe with 3 assets
        self.tickers = ["AssetA", "AssetB", "AssetC"]
        self.dates = pd.date_range("2024-01-01", periods=100)
        
        # Perfect correlation, but different volatilities
        self.returns_df = pd.DataFrame({
            "AssetA": np.random.normal(0.0005, 0.01, 100),
            "AssetB": np.random.normal(0.0008, 0.015, 100),
            "AssetC": np.random.normal(0.0003, 0.008, 100)
        }, index=self.dates)
        
    def test_mpt_optimization_constraints(self):
        # Run optimization
        res = optimize_portfolio_mpt(self.returns_df, self.tickers)
        
        # Verify maximum Sharpe weights sum to 1.0
        max_s_weights = res["max_sharpe"]["weights"]
        sum_weights = sum(max_s_weights.values())
        self.assertAlmostEqual(sum_weights, 1.0, places=4)
        
        # Verify weight boundary conditions (each must be between 0.0 and 1.0)
        for t in self.tickers:
            self.assertTrue(0.0 <= max_s_weights[t] <= 1.0001)

        # Verify minimum Variance weights sum to 1.0
        min_v_weights = res["min_variance"]["weights"]
        sum_weights_v = sum(min_v_weights.values())
        self.assertAlmostEqual(sum_weights_v, 1.0, places=4)

    def test_diversification_ratio_single_asset(self):
        # Single asset correlation/variance should yield DR of exactly 1.0
        single_cov = pd.DataFrame([[0.02]], index=["A"], columns=["A"])
        weights = [1.0]
        dr = calculate_diversification_ratio(single_cov, weights)
        self.assertAlmostEqual(dr, 1.0, places=5)

    def test_diversification_ratio_multiple(self):
        # High diversification should yield DR > 1.0
        cov_matrix = pd.DataFrame([
            [0.04, 0.00, 0.00],
            [0.00, 0.09, 0.00],
            [0.00, 0.00, 0.01]
        ], index=["A", "B", "C"], columns=["A", "B", "C"])
        weights = [0.33, 0.33, 0.33]
        dr = calculate_diversification_ratio(cov_matrix, weights)
        # For uncorrelated assets, diversification ratio must be strictly greater than 1.0
        self.assertTrue(dr > 1.0)

if __name__ == "__main__":
    unittest.main()
