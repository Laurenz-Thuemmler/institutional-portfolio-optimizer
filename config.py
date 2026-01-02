"""
Central configuration for the Portfolio Optimization Engine.

This module contains all configurable parameters including default tickers,
date ranges, and financial constants used throughout the application.
"""

from datetime import datetime, timedelta
from typing import List

# =============================================================================
# Default Stock Universe - Top 50 S&P 500 Constituents
# =============================================================================
DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "TSLA", "AVGO",
    "LLY", "UNH", "JPM", "V", "JNJ", "MA", "XOM", "COST", "PG", "HD",
    "MRK", "ABBV", "PEP", "KO", "BAC", "ORCL", "ADBE", "CSCO", "WMT", "CRM",
    "ACN", "MCD", "AMD", "INTC", "NFLX", "DIS", "CMCSA", "PFE", "TMO", "ABT",
    "LIN", "TXN", "CVX", "WFC", "DHR", "NEE", "PM", "AMGN", "QCOM",
]

# Benchmark for comparison
BENCHMARK_TICKER: str = "SPY"  # SPDR S&P 500 ETF Trust

# =============================================================================
# Date Configuration
# =============================================================================
# Default to 3 years of historical data
DEFAULT_END_DATE: datetime = datetime.now()
DEFAULT_START_DATE: datetime = DEFAULT_END_DATE - timedelta(days=3 * 365)

# Default train/test split (80% train, 20% test)
DEFAULT_SPLIT_RATIO: float = 0.8

# =============================================================================
# Financial Constants
# =============================================================================
# Risk-free rate (annualized) - approximately current US Treasury rate
RISK_FREE_RATE: float = 0.05

# Trading days per year (US market standard)
TRADING_DAYS_PER_YEAR: int = 252

# =============================================================================
# Optimization Parameters
# =============================================================================
# Number of random portfolios to generate for visualization
NUM_RANDOM_PORTFOLIOS: int = 2000

# Number of points on the efficient frontier curve
NUM_FRONTIER_POINTS: int = 50

# Optimization method for scipy.optimize.minimize
OPTIMIZATION_METHOD: str = "SLSQP"

# Maximum iterations for optimizer
MAX_ITERATIONS: int = 1000

# Convergence tolerance
OPTIMIZATION_TOLERANCE: float = 1e-10

# =============================================================================
# Data Cleaning Parameters
# =============================================================================
# Minimum percentage of valid data required for a ticker to be included
MIN_DATA_COMPLETENESS: float = 0.95

# Maximum allowed consecutive NaN values before dropping a ticker
MAX_CONSECUTIVE_NANS: int = 5
