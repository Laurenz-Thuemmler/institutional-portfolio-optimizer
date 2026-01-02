"""
Portfolio Optimization Engine - Source Package

This package contains the core modules for portfolio optimization
based on Modern Portfolio Theory (MPT).

Modules:
    - data_loader: Financial data acquisition and preprocessing
    - mathematics: Quantitative metrics and calculations
    - optimizer: Portfolio optimization algorithms
    - visualizer: Interactive chart generation
"""

from src.data_loader import FinancialDataLoader
from src.mathematics import QuantMetrics
from src.optimizer import PortfolioOptimizer
from src.visualizer import DashboardCharts

__all__ = [
    "FinancialDataLoader",
    "QuantMetrics",
    "PortfolioOptimizer",
    "DashboardCharts",
]

__version__ = "1.0.0"
