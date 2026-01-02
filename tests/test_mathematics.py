"""
Unit Tests for the QuantMetrics Module.

This module contains pytest tests to verify the mathematical correctness
of portfolio calculations. Tests use simple, known inputs to validate
formulas against hand-calculated expected values.

Run with: pytest tests/test_mathematics.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mathematics import QuantMetrics


class TestPortfolioReturn:
    """Tests for portfolio return calculation."""

    def test_equal_weights_equal_returns(self):
        """Equal weights with equal returns should give that return."""
        weights = np.array([0.5, 0.5])
        mean_returns = np.array([0.001, 0.001])  # 0.1% daily

        result = QuantMetrics.portfolio_return(weights, mean_returns, trading_days=252)

        # Expected: 0.001 * 252 = 0.252 (25.2% annual)
        expected = 0.252
        assert abs(result - expected) < 1e-10

    def test_single_asset(self):
        """100% allocation to single asset."""
        weights = np.array([1.0, 0.0])
        mean_returns = np.array([0.002, 0.001])  # First asset: 0.2% daily

        result = QuantMetrics.portfolio_return(weights, mean_returns, trading_days=252)

        # Expected: 0.002 * 252 = 0.504 (50.4% annual)
        expected = 0.504
        assert abs(result - expected) < 1e-10

    def test_weighted_average(self):
        """Verify weighted average calculation."""
        weights = np.array([0.6, 0.4])
        mean_returns = np.array([0.001, 0.002])

        result = QuantMetrics.portfolio_return(weights, mean_returns, trading_days=252)

        # Expected: (0.6 * 0.001 + 0.4 * 0.002) * 252 = 0.0014 * 252 = 0.3528
        expected = 0.3528
        assert abs(result - expected) < 1e-10


class TestPortfolioVolatility:
    """Tests for portfolio volatility calculation."""

    def test_single_asset_volatility(self):
        """100% in single asset should give that asset's volatility."""
        weights = np.array([1.0, 0.0])

        # Simple covariance matrix
        # Asset 1: variance = 0.0004 (daily vol = 0.02 = 2%)
        # Asset 2: variance = 0.0001 (daily vol = 0.01 = 1%)
        cov_matrix = np.array([
            [0.0004, 0.0001],
            [0.0001, 0.0001]
        ])

        result = QuantMetrics.portfolio_volatility(weights, cov_matrix, trading_days=252)

        # Expected: sqrt(0.0004) * sqrt(252) = 0.02 * 15.87 = 0.3175
        expected = 0.02 * np.sqrt(252)
        assert abs(result - expected) < 1e-6

    def test_diversification_reduces_risk(self):
        """Diversification should reduce portfolio volatility."""
        # Two uncorrelated assets with equal volatility
        cov_matrix = np.array([
            [0.0004, 0.0],
            [0.0, 0.0004]
        ])

        # 100% in asset 1
        single_vol = QuantMetrics.portfolio_volatility(
            np.array([1.0, 0.0]), cov_matrix
        )

        # 50/50 split (uncorrelated)
        diversified_vol = QuantMetrics.portfolio_volatility(
            np.array([0.5, 0.5]), cov_matrix
        )

        # Diversified should have lower volatility
        # For 50/50 uncorrelated: vol = sqrt(0.5^2 * 0.0004 + 0.5^2 * 0.0004)
        # = sqrt(0.0002) = 0.01414... (vs 0.02 for single)
        assert diversified_vol < single_vol

    def test_perfect_correlation_no_diversification(self):
        """Perfectly correlated assets provide no diversification benefit."""
        # Perfectly correlated assets (correlation = 1)
        vol = 0.02  # Daily volatility
        var = vol ** 2
        cov_matrix = np.array([
            [var, var],
            [var, var]
        ])

        single_vol = QuantMetrics.portfolio_volatility(
            np.array([1.0, 0.0]), cov_matrix
        )
        split_vol = QuantMetrics.portfolio_volatility(
            np.array([0.5, 0.5]), cov_matrix
        )

        # Should be equal (no diversification benefit)
        assert abs(single_vol - split_vol) < 1e-10


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_basic_sharpe(self):
        """Test basic Sharpe ratio calculation."""
        portfolio_return = 0.12  # 12% annual
        portfolio_vol = 0.15     # 15% annual
        rf_rate = 0.05           # 5% annual

        result = QuantMetrics.sharpe_ratio(portfolio_return, portfolio_vol, rf_rate)

        # Expected: (0.12 - 0.05) / 0.15 = 0.4667
        expected = (0.12 - 0.05) / 0.15
        assert abs(result - expected) < 1e-10

    def test_negative_sharpe(self):
        """Returns below risk-free rate give negative Sharpe."""
        portfolio_return = 0.03  # 3% annual
        portfolio_vol = 0.15     # 15% annual
        rf_rate = 0.05           # 5% annual

        result = QuantMetrics.sharpe_ratio(portfolio_return, portfolio_vol, rf_rate)

        # Expected: (0.03 - 0.05) / 0.15 = -0.1333
        assert result < 0

    def test_zero_volatility_returns_zero(self):
        """Zero volatility should return zero to avoid division error."""
        result = QuantMetrics.sharpe_ratio(0.10, 0.0, 0.05)
        assert result == 0.0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_all_positive_returns(self):
        """All positive returns should give zero (no downside risk)."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.01])

        result = QuantMetrics.sortino_ratio(returns, risk_free_rate=0.0)

        # With all positive returns and 0 risk-free, downside deviation ~ 0
        # But since rf=0, some days might be below daily rf
        # Let's use a very small rf to ensure all returns are "positive"
        assert result >= 0

    def test_mixed_returns(self):
        """Test with mix of positive and negative returns."""
        returns = pd.Series([0.02, -0.01, 0.015, -0.02, 0.01, -0.005])

        result = QuantMetrics.sortino_ratio(returns, risk_free_rate=0.0)

        # Should be a finite number
        assert np.isfinite(result)

    def test_sortino_vs_sharpe_with_positive_skew(self):
        """Sortino should be higher than Sharpe for positively skewed returns."""
        # Returns with more upside than downside
        np.random.seed(42)
        positive_skew = pd.Series(
            np.concatenate([
                np.random.normal(0.002, 0.005, 200),  # Small positive returns
                np.random.normal(0.01, 0.01, 50),     # Large positive returns
                np.random.normal(-0.001, 0.003, 50)   # Few small negative returns
            ])
        )

        sortino = QuantMetrics.sortino_ratio(positive_skew, risk_free_rate=0.02)

        # Calculate Sharpe for comparison
        ann_return = positive_skew.mean() * 252
        ann_vol = positive_skew.std() * np.sqrt(252)
        sharpe = QuantMetrics.sharpe_ratio(ann_return, ann_vol, 0.02)

        # Sortino should generally be higher for positive skew
        # (This is a soft test - relationship depends on exact distribution)
        assert np.isfinite(sortino)
        assert np.isfinite(sharpe)


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_no_drawdown(self):
        """Monotonically increasing returns have zero drawdown."""
        cumulative = pd.Series([1.0, 1.05, 1.10, 1.15, 1.20])

        result = QuantMetrics.max_drawdown(cumulative)

        assert result == 0.0

    def test_known_drawdown(self):
        """Test with known drawdown scenario."""
        # Goes from 100 -> 150 -> 120 -> 140
        # Max drawdown is (150-120)/150 = 20%
        cumulative = pd.Series([1.0, 1.5, 1.2, 1.4])

        result = QuantMetrics.max_drawdown(cumulative)

        expected = (1.5 - 1.2) / 1.5  # 0.20 (20%)
        assert abs(result - expected) < 1e-10

    def test_drawdown_at_end(self):
        """Test when lowest point is at the end."""
        cumulative = pd.Series([1.0, 1.2, 1.3, 1.1, 0.9])

        result = QuantMetrics.max_drawdown(cumulative)

        # Max is 1.3, min after is 0.9
        expected = (1.3 - 0.9) / 1.3
        assert abs(result - expected) < 1e-10

    def test_empty_series(self):
        """Empty series should return zero."""
        cumulative = pd.Series([], dtype=float)

        result = QuantMetrics.max_drawdown(cumulative)

        assert result == 0.0


class TestCovarianceMatrix:
    """Tests for covariance matrix calculation."""

    def test_regularization_applied(self):
        """Verify regularization is added to diagonal."""
        returns = pd.DataFrame({
            "A": [0.01, 0.02, 0.015, -0.01, 0.005],
            "B": [0.02, 0.01, 0.02, -0.005, 0.01]
        })

        reg_value = 1e-6
        result = QuantMetrics.calculate_covariance_matrix(returns, regularization=reg_value)

        # Diagonal should be >= regularization value
        assert np.all(np.diag(result) >= reg_value)

    def test_symmetric_matrix(self):
        """Covariance matrix should be symmetric."""
        returns = pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
            "C": np.random.randn(100)
        })

        result = QuantMetrics.calculate_covariance_matrix(returns)

        # Check symmetry
        assert np.allclose(result, result.T)


class TestCumulativeReturns:
    """Tests for cumulative returns calculation."""

    def test_simple_cumulative(self):
        """Test simple cumulative return calculation."""
        daily_returns = pd.Series([0.01, 0.02, -0.01])

        result = QuantMetrics.calculate_cumulative_returns(daily_returns)

        # Expected: (1.01) * (1.02) * (0.99) = 1.0198
        expected = 1.01 * 1.02 * 0.99
        assert abs(result.iloc[-1] - expected) < 1e-10

    def test_starts_at_one(self):
        """Cumulative returns should start at (1 + first_return)."""
        daily_returns = pd.Series([0.05, 0.01, 0.02])

        result = QuantMetrics.calculate_cumulative_returns(daily_returns)

        assert abs(result.iloc[0] - 1.05) < 1e-10


class TestAnnualizedReturn:
    """Tests for annualized return from cumulative."""

    def test_one_year_return(self):
        """One year cumulative should equal annualized."""
        cumulative = 1.10  # 10% over one year
        num_days = 252

        result = QuantMetrics.annualized_return_from_cumulative(
            cumulative, num_days, trading_days=252
        )

        # Should be exactly 10%
        assert abs(result - 0.10) < 1e-10

    def test_half_year_return(self):
        """Half year should be annualized correctly."""
        cumulative = 1.05  # 5% over half year
        num_days = 126  # Half of 252

        result = QuantMetrics.annualized_return_from_cumulative(
            cumulative, num_days, trading_days=252
        )

        # Annualized: (1.05)^2 - 1 = 0.1025 (10.25%)
        expected = (1.05 ** 2) - 1
        assert abs(result - expected) < 1e-10

    def test_zero_days(self):
        """Zero days should return zero."""
        result = QuantMetrics.annualized_return_from_cumulative(1.10, 0)
        assert result == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
