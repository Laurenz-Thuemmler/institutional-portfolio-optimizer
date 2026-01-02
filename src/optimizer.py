"""
Portfolio Optimization Module.

This module implements portfolio optimization algorithms based on Modern Portfolio
Theory (MPT). It uses scipy's optimization routines to find optimal portfolio
weights that either maximize the Sharpe ratio or minimize volatility.

Key Concepts:
    - Efficient Frontier: The set of portfolios offering the highest expected
      return for each level of risk.
    - Maximum Sharpe Ratio Portfolio: The portfolio with the highest risk-adjusted
      return (tangent to the Capital Market Line).
    - Minimum Volatility Portfolio: The portfolio with the lowest possible risk.

Optimization Approach:
    We use Sequential Least Squares Programming (SLSQP) to solve the constrained
    optimization problem:
    - Constraint: Sum of weights = 1 (fully invested)
    - Bounds: 0 <= weight <= max_weight for each asset (long-only, diversified)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    NUM_RANDOM_PORTFOLIOS,
    NUM_FRONTIER_POINTS,
    OPTIMIZATION_METHOD,
    MAX_ITERATIONS,
    OPTIMIZATION_TOLERANCE,
)
from src.mathematics import QuantMetrics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """
    Container for portfolio optimization results.

    Attributes:
        weights: Dictionary mapping ticker to weight.
        expected_return: Annualized expected portfolio return.
        expected_volatility: Annualized portfolio volatility.
        sharpe_ratio: Risk-adjusted return metric.
        success: Whether optimization converged successfully.
        message: Optimization status message.
    """
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    success: bool
    message: str


class PortfolioOptimizer:
    """
    Implements portfolio optimization using Modern Portfolio Theory.

    This class provides methods to find optimal portfolio allocations by
    maximizing the Sharpe ratio or minimizing volatility, subject to
    constraints on weight bounds and full investment.

    Attributes:
        returns: DataFrame of asset returns.
        tickers: List of ticker symbols.
        n_assets: Number of assets in the portfolio.
        mean_returns: Mean daily returns for each asset.
        cov_matrix: Covariance matrix of returns.
        risk_free_rate: Annual risk-free rate.

    Example:
        >>> optimizer = PortfolioOptimizer(returns_df)
        >>> result = optimizer.optimize_max_sharpe()
        >>> print(f"Optimal weights: {result.weights}")
        >>> print(f"Expected Sharpe: {result.sharpe_ratio:.2f}")
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = RISK_FREE_RATE,
        max_weight: float = 1.0
    ) -> None:
        """
        Initialize the PortfolioOptimizer.

        Args:
            returns: DataFrame of daily returns (rows=dates, columns=assets).
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            max_weight: Maximum weight allowed per asset (0.0 to 1.0).
                       E.g., 0.05 means no single asset can exceed 5% of portfolio,
                       forcing diversification across at least 20 assets.
        """
        self.returns = returns
        self.tickers: List[str] = list(returns.columns)
        self.n_assets: int = len(self.tickers)
        self.risk_free_rate: float = risk_free_rate
        self.max_weight: float = max_weight

        # Pre-compute statistics for optimization
        self.mean_returns: np.ndarray = returns.mean().values
        self.cov_matrix: np.ndarray = QuantMetrics.calculate_covariance_matrix(returns)

        # Optimization constraints and bounds
        self._constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        # Bounds enforce long-only and max weight constraint for diversification
        self._bounds = tuple((0.0, self.max_weight) for _ in range(self.n_assets))

        # Initial guess: equal weights (capped at max_weight if necessary)
        equal_weight = 1.0 / self.n_assets
        self._initial_weights = np.array([min(equal_weight, self.max_weight)] * self.n_assets)
        # Normalize to sum to 1
        self._initial_weights = self._initial_weights / self._initial_weights.sum()

    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """
        Calculate negative Sharpe ratio for minimization.

        We minimize the negative Sharpe ratio because scipy.optimize.minimize
        finds minima, but we want to maximize Sharpe.

        Args:
            weights: Array of portfolio weights.

        Returns:
            Negative Sharpe ratio (to be minimized).
        """
        portfolio_return = QuantMetrics.portfolio_return(
            weights, self.mean_returns, TRADING_DAYS_PER_YEAR
        )
        portfolio_vol = QuantMetrics.portfolio_volatility(
            weights, self.cov_matrix, TRADING_DAYS_PER_YEAR
        )
        sharpe = QuantMetrics.sharpe_ratio(
            portfolio_return, portfolio_vol, self.risk_free_rate
        )
        return -sharpe

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility for minimization.

        Args:
            weights: Array of portfolio weights.

        Returns:
            Annualized portfolio volatility.
        """
        return QuantMetrics.portfolio_volatility(
            weights, self.cov_matrix, TRADING_DAYS_PER_YEAR
        )

    def _create_result(
        self,
        weights: np.ndarray,
        success: bool,
        message: str
    ) -> OptimizationResult:
        """
        Create an OptimizationResult from weight array.

        Args:
            weights: Array of optimized weights.
            success: Whether optimization succeeded.
            message: Status message.

        Returns:
            OptimizationResult with all portfolio metrics.
        """
        expected_return = QuantMetrics.portfolio_return(
            weights, self.mean_returns, TRADING_DAYS_PER_YEAR
        )
        expected_vol = QuantMetrics.portfolio_volatility(
            weights, self.cov_matrix, TRADING_DAYS_PER_YEAR
        )
        sharpe = QuantMetrics.sharpe_ratio(
            expected_return, expected_vol, self.risk_free_rate
        )

        # Create weights dictionary
        weights_dict = {
            ticker: round(weight, 6)
            for ticker, weight in zip(self.tickers, weights)
        }

        return OptimizationResult(
            weights=weights_dict,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe_ratio=sharpe,
            success=success,
            message=message
        )

    def _get_equal_weight_fallback(self, error_msg: str) -> OptimizationResult:
        """
        Create an equal-weight portfolio as fallback when optimization fails.

        Args:
            error_msg: Error message describing why optimization failed.

        Returns:
            OptimizationResult with equal weights.
        """
        logger.warning(f"Optimization failed: {error_msg}. Using equal weights.")
        return self._create_result(
            self._initial_weights,
            success=False,
            message=f"Optimization failed: {error_msg}. Falling back to equal weights."
        )

    def optimize_max_sharpe(self) -> OptimizationResult:
        """
        Find the portfolio that maximizes the Sharpe ratio.

        This is the optimal risky portfolio - the point where the Capital Market
        Line is tangent to the Efficient Frontier.

        Returns:
            OptimizationResult with optimal weights and expected metrics.

        Note:
            If optimization fails, returns equal-weighted portfolio as fallback.
        """
        try:
            result: OptimizeResult = minimize(
                self._negative_sharpe,
                self._initial_weights,
                method=OPTIMIZATION_METHOD,
                bounds=self._bounds,
                constraints=self._constraints,
                options={
                    "maxiter": MAX_ITERATIONS,
                    "ftol": OPTIMIZATION_TOLERANCE
                }
            )

            if result.success:
                return self._create_result(
                    result.x,
                    success=True,
                    message="Optimization converged successfully."
                )
            else:
                return self._get_equal_weight_fallback(result.message)

        except Exception as e:
            return self._get_equal_weight_fallback(str(e))

    def optimize_min_volatility(self) -> OptimizationResult:
        """
        Find the portfolio that minimizes volatility (risk).

        This is the leftmost point on the Efficient Frontier - the portfolio
        with the lowest possible risk among all possible portfolios.

        Returns:
            OptimizationResult with optimal weights and expected metrics.

        Note:
            If optimization fails, returns equal-weighted portfolio as fallback.
        """
        try:
            result: OptimizeResult = minimize(
                self._portfolio_volatility,
                self._initial_weights,
                method=OPTIMIZATION_METHOD,
                bounds=self._bounds,
                constraints=self._constraints,
                options={
                    "maxiter": MAX_ITERATIONS,
                    "ftol": OPTIMIZATION_TOLERANCE
                }
            )

            if result.success:
                return self._create_result(
                    result.x,
                    success=True,
                    message="Optimization converged successfully."
                )
            else:
                return self._get_equal_weight_fallback(result.message)

        except Exception as e:
            return self._get_equal_weight_fallback(str(e))

    def optimize_target_return(self, target_return: float) -> OptimizationResult:
        """
        Find the minimum volatility portfolio for a target return.

        This is used to construct the efficient frontier by finding the
        lowest-risk portfolio at each return level.

        Args:
            target_return: Annualized target return.

        Returns:
            OptimizationResult for the target return.
        """
        # Add constraint for target return
        constraints = self._constraints.copy()
        constraints.append({
            "type": "eq",
            "fun": lambda w: QuantMetrics.portfolio_return(
                w, self.mean_returns, TRADING_DAYS_PER_YEAR
            ) - target_return
        })

        try:
            result: OptimizeResult = minimize(
                self._portfolio_volatility,
                self._initial_weights,
                method=OPTIMIZATION_METHOD,
                bounds=self._bounds,
                constraints=constraints,
                options={
                    "maxiter": MAX_ITERATIONS,
                    "ftol": OPTIMIZATION_TOLERANCE
                }
            )

            if result.success:
                return self._create_result(
                    result.x,
                    success=True,
                    message="Optimization converged successfully."
                )
            else:
                return self._get_equal_weight_fallback(result.message)

        except Exception as e:
            return self._get_equal_weight_fallback(str(e))

    def generate_random_portfolios(
        self,
        n_portfolios: int = NUM_RANDOM_PORTFOLIOS
    ) -> pd.DataFrame:
        """
        Generate random portfolio allocations for visualization.

        Creates random weight combinations that satisfy constraints
        (weights sum to 1, long-only, respecting max_weight) to visualize
        the opportunity set.

        Args:
            n_portfolios: Number of random portfolios to generate.

        Returns:
            DataFrame with columns: Return, Volatility, Sharpe, and one per ticker.
        """
        results = []

        for _ in range(n_portfolios):
            # Generate random weights using Dirichlet distribution
            # This ensures weights sum to 1 and are all positive
            weights = np.random.dirichlet(np.ones(self.n_assets))

            # Cap weights at max_weight and redistribute excess
            if self.max_weight < 1.0:
                weights = self._cap_weights(weights)

            portfolio_return = QuantMetrics.portfolio_return(
                weights, self.mean_returns, TRADING_DAYS_PER_YEAR
            )
            portfolio_vol = QuantMetrics.portfolio_volatility(
                weights, self.cov_matrix, TRADING_DAYS_PER_YEAR
            )
            sharpe = QuantMetrics.sharpe_ratio(
                portfolio_return, portfolio_vol, self.risk_free_rate
            )

            result = {
                "Return": portfolio_return,
                "Volatility": portfolio_vol,
                "Sharpe": sharpe
            }
            for ticker, weight in zip(self.tickers, weights):
                result[ticker] = weight

            results.append(result)

        return pd.DataFrame(results)

    def _cap_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Cap weights at max_weight and redistribute excess proportionally.

        Args:
            weights: Array of portfolio weights summing to 1.

        Returns:
            Adjusted weights respecting max_weight constraint.
        """
        capped = np.minimum(weights, self.max_weight)
        excess = 1.0 - capped.sum()

        # Redistribute excess to uncapped weights
        iterations = 0
        while excess > 1e-10 and iterations < 100:
            uncapped_mask = capped < self.max_weight
            if not uncapped_mask.any():
                break
            uncapped_weights = capped[uncapped_mask]
            addition = excess * (uncapped_weights / uncapped_weights.sum())
            capped[uncapped_mask] += addition
            capped = np.minimum(capped, self.max_weight)
            excess = 1.0 - capped.sum()
            iterations += 1

        # Final normalization to ensure sum = 1
        return capped / capped.sum()

    def generate_efficient_frontier(
        self,
        n_points: int = NUM_FRONTIER_POINTS
    ) -> pd.DataFrame:
        """
        Generate points along the efficient frontier.

        The efficient frontier is the set of optimal portfolios that offer
        the highest expected return for a defined level of risk.

        Args:
            n_points: Number of points to generate along the frontier.

        Returns:
            DataFrame with columns: Return, Volatility, Sharpe.
        """
        # Get min and max possible returns
        min_vol_result = self.optimize_min_volatility()
        max_sharpe_result = self.optimize_max_sharpe()

        min_return = min_vol_result.expected_return
        max_return = max(
            max_sharpe_result.expected_return,
            self.mean_returns.max() * TRADING_DAYS_PER_YEAR
        )

        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)

        frontier_points = []
        for target in target_returns:
            result = self.optimize_target_return(target)
            if result.success:
                frontier_points.append({
                    "Return": result.expected_return,
                    "Volatility": result.expected_volatility,
                    "Sharpe": result.sharpe_ratio
                })

        return pd.DataFrame(frontier_points)

    def get_equal_weight_portfolio(self) -> OptimizationResult:
        """
        Calculate metrics for an equal-weighted portfolio.

        Returns:
            OptimizationResult with equal weights.
        """
        return self._create_result(
            self._initial_weights,
            success=True,
            message="Equal weight portfolio."
        )

    def calculate_portfolio_returns(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate portfolio returns given weights and return data.

        Args:
            weights: Dictionary of {ticker: weight}.
            returns: DataFrame of daily returns.

        Returns:
            Series of daily portfolio returns.
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in returns.columns])
        return returns.dot(weights_array)
