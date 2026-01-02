"""
Financial Data Loader Module.

This module handles the acquisition, cleaning, and preprocessing of financial
market data using the yfinance library. It provides robust error handling
and data quality checks to ensure reliable inputs for portfolio optimization.

Features:
    - Download adjusted close prices from Yahoo Finance
    - Calculate log returns for statistical accuracy
    - Train/test data splitting for backtesting
    - Automatic handling of missing data and failed downloads
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_TICKERS,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    BENCHMARK_TICKER,
    MIN_DATA_COMPLETENESS,
    MAX_CONSECUTIVE_NANS,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataLoader:
    """
    Handles downloading and preprocessing of financial market data.

    This class provides a clean interface for fetching historical price data,
    calculating returns, and preparing data for portfolio optimization.

    Attributes:
        tickers: List of stock ticker symbols.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        prices: DataFrame of adjusted close prices.
        returns: DataFrame of log returns.
        failed_tickers: List of tickers that failed to download.

    Example:
        >>> loader = FinancialDataLoader(["AAPL", "MSFT", "GOOGL"])
        >>> loader.download_data()
        >>> train, test = loader.get_train_test_split("2023-01-01")
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_benchmark: bool = True
    ) -> None:
        """
        Initialize the FinancialDataLoader.

        Args:
            tickers: List of ticker symbols. Defaults to config DEFAULT_TICKERS.
            start_date: Start date for data. Defaults to config DEFAULT_START_DATE.
            end_date: End date for data. Defaults to config DEFAULT_END_DATE.
            include_benchmark: Whether to include SPY benchmark data.
        """
        self.tickers: List[str] = tickers if tickers else DEFAULT_TICKERS.copy()
        self.start_date: datetime = start_date if start_date else DEFAULT_START_DATE
        self.end_date: datetime = end_date if end_date else DEFAULT_END_DATE
        self.include_benchmark: bool = include_benchmark

        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.benchmark_prices: Optional[pd.Series] = None
        self.benchmark_returns: Optional[pd.Series] = None
        self.failed_tickers: List[str] = []
        self.valid_tickers: List[str] = []

    def download_data(self) -> pd.DataFrame:
        """
        Download adjusted close prices for all tickers.

        Fetches historical data from Yahoo Finance with robust error handling.
        Failed tickers are logged and excluded from the dataset.

        Returns:
            DataFrame of adjusted close prices (rows=dates, columns=tickers).

        Raises:
            ValueError: If no valid data could be downloaded.
        """
        logger.info(f"Downloading data for {len(self.tickers)} tickers...")

        all_data: Dict[str, pd.Series] = {}
        self.failed_tickers = []

        for ticker in self.tickers:
            try:
                data = self._download_single_ticker(ticker)
                if data is not None and len(data) > 0:
                    all_data[ticker] = data
                    logger.info(f"Successfully downloaded {ticker}")
                else:
                    self.failed_tickers.append(ticker)
                    logger.warning(f"No data available for {ticker}")
            except Exception as e:
                self.failed_tickers.append(ticker)
                logger.warning(f"Failed to download {ticker}: {str(e)}")

        if not all_data:
            raise ValueError("Failed to download data for any tickers.")

        # Combine into DataFrame
        self.prices = pd.DataFrame(all_data)
        self.valid_tickers = list(self.prices.columns)

        # Clean the data
        self._clean_data()

        # Calculate returns
        self.returns = self._calculate_log_returns(self.prices)

        # Download benchmark if requested
        if self.include_benchmark:
            self._download_benchmark()

        logger.info(
            f"Successfully loaded {len(self.valid_tickers)} tickers. "
            f"Failed: {self.failed_tickers}"
        )

        return self.prices

    def _download_single_ticker(self, ticker: str) -> Optional[pd.Series]:
        """
        Download data for a single ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Series of adjusted close prices, or None if download failed.
        """
        stock = yf.Ticker(ticker)
        hist = stock.history(
            start=self.start_date.strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
            auto_adjust=True  # Use adjusted prices
        )

        if hist.empty:
            return None

        return hist["Close"]

    def _download_benchmark(self) -> None:
        """Download benchmark (SPY) data for comparison."""
        try:
            benchmark_data = self._download_single_ticker(BENCHMARK_TICKER)
            if benchmark_data is not None:
                # Remove timezone info to match portfolio data
                if benchmark_data.index.tz is not None:
                    benchmark_data.index = benchmark_data.index.tz_localize(None)
                # Align benchmark with portfolio data
                common_dates = self.prices.index.intersection(benchmark_data.index)
                self.benchmark_prices = benchmark_data.loc[common_dates]
                self.benchmark_returns = self._calculate_log_returns(
                    self.benchmark_prices.to_frame()
                ).squeeze()
                logger.info(f"Successfully downloaded benchmark: {BENCHMARK_TICKER}")
            else:
                logger.warning(f"Failed to download benchmark: {BENCHMARK_TICKER}")
        except Exception as e:
            logger.warning(f"Error downloading benchmark: {str(e)}")

    def _clean_data(self) -> None:
        """
        Clean the price data by handling missing values.

        Applies the following cleaning steps:
        1. Remove timezone info (for consistent date comparisons globally)
        2. Forward fill small gaps (up to MAX_CONSECUTIVE_NANS)
        3. Remove tickers with too many missing values
        4. Drop any remaining rows with NaN values
        """
        if self.prices is None:
            return

        # Remove timezone info for consistent date comparisons
        # This ensures the app works the same way regardless of user's timezone
        if self.prices.index.tz is not None:
            self.prices.index = self.prices.index.tz_localize(None)

        # Forward fill small gaps (weekends, holidays)
        self.prices = self.prices.ffill(limit=MAX_CONSECUTIVE_NANS)

        # Check data completeness for each ticker
        tickers_to_remove = []
        for ticker in self.prices.columns:
            completeness = self.prices[ticker].notna().mean()
            if completeness < MIN_DATA_COMPLETENESS:
                tickers_to_remove.append(ticker)
                logger.warning(
                    f"Removing {ticker}: only {completeness:.1%} complete data"
                )

        # Remove incomplete tickers
        if tickers_to_remove:
            self.prices = self.prices.drop(columns=tickers_to_remove)
            self.failed_tickers.extend(tickers_to_remove)
            self.valid_tickers = [
                t for t in self.valid_tickers if t not in tickers_to_remove
            ]

        # Drop any remaining rows with NaN
        self.prices = self.prices.dropna()

        if self.prices.empty:
            raise ValueError("No valid data remaining after cleaning.")

    def _calculate_log_returns(
        self,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate log returns from price data.

        Log returns are preferred for financial analysis because:
        1. They are additive across time (can sum daily returns)
        2. They are approximately normally distributed
        3. They handle compounding correctly

        Formula: r_t = ln(P_t / P_{t-1})

        Args:
            prices: DataFrame of price data.

        Returns:
            DataFrame of log returns (first row is NaN and dropped).
        """
        log_returns = np.log(prices / prices.shift(1))
        return log_returns.dropna()

    def get_train_test_split(
        self,
        split_date: Optional[str] = None,
        split_ratio: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split returns data into training and testing sets.

        The training set is used for portfolio optimization, while the
        test set is used for out-of-sample backtesting to evaluate
        how well the optimized portfolio performs on unseen data.

        Args:
            split_date: Date string (YYYY-MM-DD) to split on.
                       Data before this date is training, after is testing.
            split_ratio: Alternative to split_date. Fraction of data for training.
                        E.g., 0.8 means 80% training, 20% testing.

        Returns:
            Tuple of (train_returns, test_returns) DataFrames.

        Raises:
            ValueError: If returns have not been calculated yet.
        """
        if self.returns is None:
            raise ValueError("Must call download_data() before splitting.")

        if split_date is not None:
            split_dt = pd.to_datetime(split_date)
            train_returns = self.returns[self.returns.index < split_dt]
            test_returns = self.returns[self.returns.index >= split_dt]
        elif split_ratio is not None:
            split_idx = int(len(self.returns) * split_ratio)
            train_returns = self.returns.iloc[:split_idx]
            test_returns = self.returns.iloc[split_idx:]
        else:
            # Default: 80/20 split
            split_idx = int(len(self.returns) * 0.8)
            train_returns = self.returns.iloc[:split_idx]
            test_returns = self.returns.iloc[split_idx:]

        logger.info(
            f"Train/Test split: {len(train_returns)} / {len(test_returns)} days"
        )

        return train_returns, test_returns

    def get_benchmark_split(
        self,
        split_date: Optional[str] = None,
        split_ratio: Optional[float] = None
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        Split benchmark returns into training and testing sets.

        Args:
            split_date: Date string (YYYY-MM-DD) to split on.
            split_ratio: Fraction of data for training.

        Returns:
            Tuple of (train_benchmark, test_benchmark) Series.
        """
        if self.benchmark_returns is None:
            return None, None

        if split_date is not None:
            split_dt = pd.to_datetime(split_date)
            train_benchmark = self.benchmark_returns[
                self.benchmark_returns.index < split_dt
            ]
            test_benchmark = self.benchmark_returns[
                self.benchmark_returns.index >= split_dt
            ]
        elif split_ratio is not None:
            split_idx = int(len(self.benchmark_returns) * split_ratio)
            train_benchmark = self.benchmark_returns.iloc[:split_idx]
            test_benchmark = self.benchmark_returns.iloc[split_idx:]
        else:
            split_idx = int(len(self.benchmark_returns) * 0.8)
            train_benchmark = self.benchmark_returns.iloc[:split_idx]
            test_benchmark = self.benchmark_returns.iloc[split_idx:]

        return train_benchmark, test_benchmark

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Calculate summary statistics for each asset.

        Returns:
            DataFrame with annualized return, volatility, and Sharpe ratio
            for each ticker.
        """
        if self.returns is None:
            raise ValueError("Must call download_data() first.")

        from src.mathematics import QuantMetrics

        stats = []
        for ticker in self.returns.columns:
            ticker_returns = self.returns[ticker]
            ann_return = ticker_returns.mean() * 252
            ann_vol = ticker_returns.std() * np.sqrt(252)
            sharpe = QuantMetrics.sharpe_ratio(ann_return, ann_vol)

            stats.append({
                "Ticker": ticker,
                "Annual Return": ann_return,
                "Annual Volatility": ann_vol,
                "Sharpe Ratio": sharpe
            })

        return pd.DataFrame(stats).set_index("Ticker")
