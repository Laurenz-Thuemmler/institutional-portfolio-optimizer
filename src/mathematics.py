"""
Quantitative Metrics Module for Portfolio Optimization.

This module provides mathematical calculations used in Modern Portfolio Theory (MPT),
including portfolio return, volatility, Sharpe ratio, Sortino ratio, and maximum drawdown.

Mathematical Background:
------------------------
Modern Portfolio Theory, developed by Harry Markowitz in 1952, is based on the idea
that investors can construct portfolios to maximize expected return for a given level
of risk. The key insight is that an asset's risk and return should not be assessed
alone, but by how it contributes to a portfolio's overall risk and return.

Key Formulas:
    - Portfolio Return: R_p = Σ(w_i * r_i)
    - Portfolio Volatility: σ_p = √(w^T * Σ * w)
    - Sharpe Ratio: SR = (R_p - R_f) / σ_p
    - Sortino Ratio: SoR = (R_p - R_f) / σ_downside
"""

import numpy as np
import pandas as pd
from typing import Union

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE


class QuantMetrics:
    """
    A collection of static methods for calculating quantitative financial metrics.

    This class provides the mathematical foundation for portfolio optimization,
    implementing core MPT metrics with proper annualization and risk adjustments.

    All methods are static to allow for easy testing and standalone usage.
    """

    @staticmethod
    def portfolio_return(
        weights: np.ndarray,
        mean_returns: np.ndarray,
        trading_days: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate the annualized expected portfolio return.

        The portfolio return is the weighted sum of individual asset returns,
        annualized by multiplying by the number of trading days.

        Formula: R_p = Σ(w_i * r_i) * trading_days

        Args:
            weights: Array of portfolio weights (must sum to 1).
            mean_returns: Array of mean daily returns for each asset.
            trading_days: Number of trading days per year (default: 252).

        Returns:
            Annualized expected portfolio return as a decimal (e.g., 0.12 = 12%).

        Example:
            >>> weights = np.array([0.5, 0.5])
            >>> mean_returns = np.array([0.001, 0.002])
            >>> QuantMetrics.portfolio_return(weights, mean_returns)
            0.378  # 37.8% annualized return
        """
        daily_return = np.dot(weights, mean_returns)
        return daily_return * trading_days

    @staticmethod
    def portfolio_volatility(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        trading_days: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate the annualized portfolio volatility (standard deviation).

        Portfolio volatility accounts for the correlations between assets,
        which is why diversification can reduce overall portfolio risk.

        Formula: σ_p = √(w^T * Σ * w) * √(trading_days)

        Where:
            - w is the weight vector
            - Σ is the covariance matrix of returns
            - The √(trading_days) factor annualizes daily volatility

        Args:
            weights: Array of portfolio weights (must sum to 1).
            cov_matrix: Covariance matrix of daily returns (n x n).
            trading_days: Number of trading days per year (default: 252).

        Returns:
            Annualized portfolio volatility as a decimal (e.g., 0.15 = 15%).

        Raises:
            ValueError: If covariance matrix dimensions don't match weights.
        """
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        daily_volatility = np.sqrt(portfolio_variance)
        return daily_volatility * np.sqrt(trading_days)

    @staticmethod
    def sharpe_ratio(
        portfolio_return: float,
        portfolio_volatility: float,
        risk_free_rate: float = RISK_FREE_RATE
    ) -> float:
        """
        Calculate the Sharpe Ratio of a portfolio.

        The Sharpe Ratio measures risk-adjusted return, showing how much
        excess return (above risk-free rate) is earned per unit of volatility.

        Formula: SR = (R_p - R_f) / σ_p

        Interpretation:
            - SR > 1.0: Good risk-adjusted performance
            - SR > 2.0: Very good
            - SR > 3.0: Excellent

        Args:
            portfolio_return: Annualized portfolio return (decimal).
            portfolio_volatility: Annualized portfolio volatility (decimal).
            risk_free_rate: Annualized risk-free rate (decimal, default: 5%).

        Returns:
            Sharpe Ratio (dimensionless).

        Note:
            Returns 0.0 if volatility is zero or very small to avoid division errors.
        """
        if portfolio_volatility < 1e-10:
            return 0.0
        return (portfolio_return - risk_free_rate) / portfolio_volatility

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = RISK_FREE_RATE,
        trading_days: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate the Sortino Ratio of a portfolio.

        The Sortino Ratio is similar to the Sharpe Ratio but uses downside
        deviation instead of total volatility, penalizing only negative returns.
        This is more appropriate when return distributions are asymmetric.

        Formula: SoR = (R_p - R_f) / σ_downside

        Where σ_downside = √(Σ(min(r_i - target, 0)^2) / n)

        Args:
            returns: Series of daily returns.
            risk_free_rate: Annualized risk-free rate (decimal).
            trading_days: Number of trading days per year.

        Returns:
            Sortino Ratio (dimensionless).

        Note:
            Returns 0.0 if there are no negative returns (no downside risk).
        """
        # Daily target return (risk-free rate converted to daily)
        daily_rf = risk_free_rate / trading_days

        # Calculate downside returns (returns below target)
        excess_returns = returns - daily_rf
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return 0.0

        # Downside deviation (annualized)
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(trading_days)

        if downside_deviation < 1e-10:
            return 0.0

        # Annualized return
        annualized_return = returns.mean() * trading_days

        return (annualized_return - risk_free_rate) / downside_deviation

    @staticmethod
    def max_drawdown(cumulative_returns: pd.Series) -> float:
        """
        Calculate the Maximum Drawdown of a portfolio.

        Maximum Drawdown measures the largest peak-to-trough decline in portfolio
        value, representing the worst-case loss an investor could have experienced.

        Formula: MDD = max((Peak_i - Trough_i) / Peak_i)

        Args:
            cumulative_returns: Series of cumulative returns (1 + total return).
                               E.g., 1.10 means 10% total gain from start.

        Returns:
            Maximum drawdown as a positive decimal (e.g., 0.20 = 20% max loss).

        Example:
            If portfolio grew from $100 to $150 then fell to $120:
            Peak = 150, Trough = 120
            Drawdown = (150 - 120) / 150 = 0.20 or 20%
        """
        if len(cumulative_returns) == 0:
            return 0.0

        # Calculate running maximum (peak)
        running_max = cumulative_returns.cummax()

        # Calculate drawdown at each point
        drawdowns = (running_max - cumulative_returns) / running_max

        # Return maximum drawdown
        return drawdowns.max()

    @staticmethod
    def calculate_covariance_matrix(
        returns: pd.DataFrame,
        regularization: float = 1e-8
    ) -> np.ndarray:
        """
        Calculate the covariance matrix of returns with optional regularization.

        Adds a small value to the diagonal to ensure the matrix is positive
        semi-definite, which is required for optimization algorithms.

        This handles cases where the covariance matrix might be singular
        (non-invertible) due to perfectly correlated assets or insufficient data.

        Args:
            returns: DataFrame of daily returns (rows=dates, columns=assets).
            regularization: Small value added to diagonal for numerical stability.

        Returns:
            Regularized covariance matrix as numpy array.
        """
        cov_matrix = returns.cov().values

        # Add regularization to diagonal for numerical stability
        cov_matrix += np.eye(cov_matrix.shape[0]) * regularization

        return cov_matrix

    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns from a series of daily returns.

        Formula: Cumulative = Π(1 + r_i) = (1 + r_1) * (1 + r_2) * ... * (1 + r_n)

        Args:
            returns: Series of daily returns (as decimals, e.g., 0.01 = 1%).

        Returns:
            Series of cumulative returns starting at 1.0.
        """
        return (1 + returns).cumprod()

    @staticmethod
    def annualized_return_from_cumulative(
        cumulative_return: float,
        num_days: int,
        trading_days: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate annualized return from a cumulative return over a period.

        Formula: R_annual = (Cumulative)^(trading_days / num_days) - 1

        Args:
            cumulative_return: Total cumulative return (e.g., 1.25 = 25% total gain).
            num_days: Number of trading days in the period.
            trading_days: Number of trading days per year.

        Returns:
            Annualized return as a decimal.
        """
        if num_days <= 0:
            return 0.0
        return (cumulative_return ** (trading_days / num_days)) - 1
