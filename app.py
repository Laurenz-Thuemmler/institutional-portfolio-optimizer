"""
Portfolio Optimization Dashboard - Streamlit Application.

This is the main entry point for the interactive portfolio optimization dashboard.
It provides a user-friendly interface for:
    - Selecting assets and date ranges
    - Running portfolio optimization
    - Visualizing the efficient frontier
    - Backtesting optimized portfolios

Run with: streamlit run app.py
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DEFAULT_TICKERS,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    BENCHMARK_TICKER,
)
from src.data_loader import FinancialDataLoader
from src.mathematics import QuantMetrics
from src.optimizer import PortfolioOptimizer, OptimizationResult
from src.visualizer import DashboardCharts

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "optimization_run" not in st.session_state:
        st.session_state.optimization_run = False
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False


def render_sidebar() -> Dict:
    """
    Render the sidebar with input controls.

    Returns:
        Dictionary of user inputs.
    """
    st.sidebar.markdown("## Configuration")

    # Ticker selection
    st.sidebar.markdown("### Asset Selection")

    # Initialize session state for selected tickers (default: all S&P 500 Top 50)
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = DEFAULT_TICKERS.copy()

    # Initialize session state for custom tickers added by user
    if "custom_tickers" not in st.session_state:
        st.session_state.custom_tickers = []

    # All available tickers: S&P 500 Top 50 + any custom ones added by user
    all_tickers = list(set(DEFAULT_TICKERS + st.session_state.custom_tickers))

    # Callback to update selected tickers
    def on_ticker_change():
        st.session_state.selected_tickers = st.session_state.ticker_multiselect

    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=sorted(all_tickers),
        default=st.session_state.selected_tickers,
        key="ticker_multiselect",
        on_change=on_ticker_change,
        help="Select at least 2 tickers for portfolio optimization"
    )

    # Custom ticker input - adds to options AND selects them
    def add_custom_tickers():
        if st.session_state.custom_ticker_input:
            new_tickers = [
                t.strip().upper()
                for t in st.session_state.custom_ticker_input.split(",")
                if t.strip()
            ]
            # Add to custom tickers list (for dropdown options)
            st.session_state.custom_tickers = list(
                set(st.session_state.custom_tickers + new_tickers)
            )
            # Also select them
            st.session_state.selected_tickers = list(
                set(st.session_state.selected_tickers + new_tickers)
            )
            st.session_state.custom_ticker_input = ""

    st.sidebar.text_input(
        "Add Custom Tickers (comma-separated)",
        key="custom_ticker_input",
        placeholder="e.g., UBER, SQ, SHOP",
        help="Type ticker symbols and press Enter to add them",
        on_change=add_custom_tickers
    )

    # Show custom tickers if any were added
    if st.session_state.custom_tickers:
        st.sidebar.caption(f"Custom: {', '.join(sorted(st.session_state.custom_tickers))}")

    # Use session state tickers (includes any just-added custom ones)
    selected_tickers = st.session_state.selected_tickers

    # Date range
    st.sidebar.markdown("### Date Range")

    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=3*365),
        max_value=end_date
    )

    # Train/Test split
    st.sidebar.markdown("### Train/Test Split")

    total_days = (end_date - start_date).days
    default_split_days = int(total_days * 0.8)
    split_date = start_date + timedelta(days=default_split_days)

    split_date = st.sidebar.date_input(
        "Split Date",
        value=split_date,
        min_value=start_date + timedelta(days=30),
        max_value=end_date - timedelta(days=30),
        help="Data before this date is used for optimization, after for backtesting"
    )

    # Risk-free rate
    st.sidebar.markdown("### Risk Parameters")

    risk_free_rate = st.sidebar.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=RISK_FREE_RATE * 100,
        step=0.25,
        help="Annual risk-free rate (e.g., Treasury yield)"
    ) / 100

    # Max allocation constraint
    st.sidebar.markdown("### Diversification Controls")

    # Calculate minimum possible max_weight based on number of selected tickers
    num_tickers = len(selected_tickers)
    min_max_weight = max(1.0, 100.0 / num_tickers) if num_tickers > 0 else 1.0

    max_weight_pct = st.sidebar.slider(
        "Max Allocation per Asset (%)",
        min_value=min_max_weight,
        max_value=100.0,
        value=20.0,
        step=1.0,
        help=f"Maximum weight per asset. Lower values force diversification. "
             f"With {num_tickers} assets, minimum is {min_max_weight:.1f}% "
             f"(requires at least {int(100/min_max_weight)} assets)."
    )
    max_weight = max_weight_pct / 100.0

    # Show info about diversification effect
    if max_weight < 1.0:
        min_assets = int(np.ceil(1.0 / max_weight))
        st.sidebar.caption(f"Forces investment in at least {min_assets} assets")

    # Run button
    st.sidebar.markdown("---")
    run_optimization = st.sidebar.button(
        "Run Optimization",
        type="primary",
        use_container_width=True
    )

    return {
        "tickers": selected_tickers,
        "start_date": datetime.combine(start_date, datetime.min.time()),
        "end_date": datetime.combine(end_date, datetime.min.time()),
        "split_date": split_date.strftime("%Y-%m-%d"),
        "risk_free_rate": risk_free_rate,
        "max_weight": max_weight,
        "run_optimization": run_optimization
    }


def render_header() -> None:
    """Render the main header."""
    st.markdown('<p class="main-header">Portfolio Optimization Engine</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Modern Portfolio Theory | Efficient Frontier Analysis | Backtesting</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_metrics_row(
    max_sharpe: OptimizationResult,
    min_vol: OptimizationResult,
    equal_weight: OptimizationResult
) -> None:
    """Render key metrics in a row of cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Max Sharpe Ratio",
            f"{max_sharpe.sharpe_ratio:.2f}",
            help="Sharpe ratio of the optimal portfolio"
        )

    with col2:
        st.metric(
            "Expected Return",
            f"{max_sharpe.expected_return*100:.1f}%",
            help="Annualized expected return"
        )

    with col3:
        st.metric(
            "Expected Volatility",
            f"{max_sharpe.expected_volatility*100:.1f}%",
            help="Annualized volatility"
        )

    with col4:
        st.metric(
            "Min Volatility",
            f"{min_vol.expected_volatility*100:.1f}%",
            help="Lowest possible portfolio volatility"
        )

    with col5:
        st.metric(
            "Equal Weight Sharpe",
            f"{equal_weight.sharpe_ratio:.2f}",
            help="Sharpe ratio of equal-weighted portfolio"
        )


def run_backtest(
    weights: Dict[str, float],
    test_returns: pd.DataFrame,
    benchmark_returns: Optional[pd.Series]
) -> Dict:
    """
    Run backtest on test data.

    Args:
        weights: Optimized portfolio weights.
        test_returns: Test period returns.
        benchmark_returns: Benchmark returns for comparison.

    Returns:
        Dictionary with backtest results.
    """
    # Calculate portfolio returns
    weight_array = [weights.get(col, 0) for col in test_returns.columns]
    portfolio_returns = test_returns.dot(weight_array)

    # Equal weight returns
    equal_weights = [1/len(test_returns.columns)] * len(test_returns.columns)
    equal_weight_returns = test_returns.dot(equal_weights)

    # Cumulative returns
    portfolio_cumulative = QuantMetrics.calculate_cumulative_returns(portfolio_returns)
    equal_weight_cumulative = QuantMetrics.calculate_cumulative_returns(equal_weight_returns)

    benchmark_cumulative = None
    if benchmark_returns is not None:
        # Align benchmark with test returns
        common_idx = test_returns.index.intersection(benchmark_returns.index)
        if len(common_idx) > 0:
            aligned_benchmark = benchmark_returns.loc[common_idx]
            benchmark_cumulative = QuantMetrics.calculate_cumulative_returns(aligned_benchmark)

    # Calculate metrics
    num_days = len(test_returns)

    portfolio_metrics = {
        "Total Return": portfolio_cumulative.iloc[-1] - 1,
        "Annual Return": QuantMetrics.annualized_return_from_cumulative(
            portfolio_cumulative.iloc[-1], num_days
        ),
        "Annual Volatility": portfolio_returns.std() * (TRADING_DAYS_PER_YEAR ** 0.5),
        "Max Drawdown": QuantMetrics.max_drawdown(portfolio_cumulative),
        "Sharpe Ratio": 0,
        "Sortino Ratio": QuantMetrics.sortino_ratio(portfolio_returns)
    }
    portfolio_metrics["Sharpe Ratio"] = QuantMetrics.sharpe_ratio(
        portfolio_metrics["Annual Return"],
        portfolio_metrics["Annual Volatility"]
    )

    equal_weight_metrics = {
        "Total Return": equal_weight_cumulative.iloc[-1] - 1,
        "Annual Return": QuantMetrics.annualized_return_from_cumulative(
            equal_weight_cumulative.iloc[-1], num_days
        ),
        "Annual Volatility": equal_weight_returns.std() * (TRADING_DAYS_PER_YEAR ** 0.5),
        "Max Drawdown": QuantMetrics.max_drawdown(equal_weight_cumulative),
        "Sharpe Ratio": 0,
        "Sortino Ratio": QuantMetrics.sortino_ratio(equal_weight_returns)
    }
    equal_weight_metrics["Sharpe Ratio"] = QuantMetrics.sharpe_ratio(
        equal_weight_metrics["Annual Return"],
        equal_weight_metrics["Annual Volatility"]
    )

    benchmark_metrics = None
    if benchmark_cumulative is not None and len(benchmark_cumulative) > 0:
        benchmark_aligned = benchmark_returns.loc[common_idx]
        benchmark_metrics = {
            "Total Return": benchmark_cumulative.iloc[-1] - 1,
            "Annual Return": QuantMetrics.annualized_return_from_cumulative(
                benchmark_cumulative.iloc[-1], len(benchmark_cumulative)
            ),
            "Annual Volatility": benchmark_aligned.std() * (TRADING_DAYS_PER_YEAR ** 0.5),
            "Max Drawdown": QuantMetrics.max_drawdown(benchmark_cumulative),
            "Sharpe Ratio": 0,
            "Sortino Ratio": QuantMetrics.sortino_ratio(benchmark_aligned)
        }
        benchmark_metrics["Sharpe Ratio"] = QuantMetrics.sharpe_ratio(
            benchmark_metrics["Annual Return"],
            benchmark_metrics["Annual Volatility"]
        )

    return {
        "portfolio_cumulative": portfolio_cumulative,
        "equal_weight_cumulative": equal_weight_cumulative,
        "benchmark_cumulative": benchmark_cumulative,
        "portfolio_metrics": portfolio_metrics,
        "equal_weight_metrics": equal_weight_metrics,
        "benchmark_metrics": benchmark_metrics
    }


def main() -> None:
    """Main application function."""
    initialize_session_state()
    render_header()

    # Sidebar inputs
    inputs = render_sidebar()

    # Validation
    if len(inputs["tickers"]) < 2:
        st.warning("Please select at least 2 tickers to run optimization.")
        return

    # Run optimization when button clicked
    if inputs["run_optimization"]:
        with st.spinner("Downloading data and running optimization..."):
            st.caption("This may take up to 1–2 minutes.")
            try:
                # Load data
                loader = FinancialDataLoader(
                    tickers=inputs["tickers"],
                    start_date=inputs["start_date"],
                    end_date=inputs["end_date"],
                    include_benchmark=True
                )
                loader.download_data()

                # Check if we have enough data
                if loader.returns is None or len(loader.returns) < 30:
                    st.error("Insufficient data. Please try different tickers or date range.")
                    return

                # Split data
                train_returns, test_returns = loader.get_train_test_split(
                    split_date=inputs["split_date"]
                )
                _, test_benchmark = loader.get_benchmark_split(
                    split_date=inputs["split_date"]
                )

                # Run optimization
                optimizer = PortfolioOptimizer(
                    train_returns,
                    risk_free_rate=inputs["risk_free_rate"],
                    max_weight=inputs["max_weight"]
                )

                max_sharpe = optimizer.optimize_max_sharpe()
                min_vol = optimizer.optimize_min_volatility()
                equal_weight = optimizer.get_equal_weight_portfolio()

                # Generate visualization data
                random_portfolios = optimizer.generate_random_portfolios(2000)
                frontier_points = optimizer.generate_efficient_frontier(50)

                # Run backtest
                backtest_results = run_backtest(
                    max_sharpe.weights,
                    test_returns,
                    test_benchmark
                )

                # Store in session state
                st.session_state.loader = loader
                st.session_state.optimizer = optimizer
                st.session_state.max_sharpe = max_sharpe
                st.session_state.min_vol = min_vol
                st.session_state.equal_weight = equal_weight
                st.session_state.random_portfolios = random_portfolios
                st.session_state.frontier_points = frontier_points
                st.session_state.train_returns = train_returns
                st.session_state.test_returns = test_returns
                st.session_state.backtest_results = backtest_results
                st.session_state.optimization_run = True

                if loader.failed_tickers:
                    st.warning(f"Some tickers failed to load: {loader.failed_tickers}")

            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                return

    # Display results if optimization has been run
    if st.session_state.optimization_run:
        max_sharpe = st.session_state.max_sharpe
        min_vol = st.session_state.min_vol
        equal_weight = st.session_state.equal_weight

        # Show warning if optimization didn't converge
        if not max_sharpe.success:
            st.warning(max_sharpe.message)

        # Metrics row
        render_metrics_row(max_sharpe, min_vol, equal_weight)
        st.markdown("---")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Optimization", "Backtest", "Metrics"])

        with tab1:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Efficient Frontier
                fig = DashboardCharts.plot_efficient_frontier(
                    st.session_state.random_portfolios,
                    st.session_state.frontier_points,
                    {
                        "expected_return": max_sharpe.expected_return,
                        "expected_volatility": max_sharpe.expected_volatility,
                        "sharpe_ratio": max_sharpe.sharpe_ratio
                    },
                    {
                        "expected_return": min_vol.expected_return,
                        "expected_volatility": min_vol.expected_volatility,
                        "sharpe_ratio": min_vol.sharpe_ratio
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Allocation donut
                fig = DashboardCharts.plot_allocation_donut(max_sharpe.weights)
                st.plotly_chart(fig, use_container_width=True)

                # Weights table
                st.markdown("### Optimal Weights")
                weights_df = pd.DataFrame([
                    {"Ticker": k, "Weight": f"{v*100:.2f}%"}
                    for k, v in sorted(
                        max_sharpe.weights.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    if v > 0.001
                ])
                st.dataframe(weights_df, hide_index=True, use_container_width=True)

        with tab2:
            backtest = st.session_state.backtest_results

            # Performance chart
            fig = DashboardCharts.plot_backtest_performance(
                backtest["portfolio_cumulative"],
                backtest["equal_weight_cumulative"],
                backtest["benchmark_cumulative"]
            )
            st.plotly_chart(fig, use_container_width=True)

            # Performance comparison table
            st.markdown("### Performance Comparison (Test Period)")

            metrics_data = {
                "Metric": ["Total Return", "Annual Return", "Annual Volatility",
                          "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"],
                "Optimized": [
                    f"{backtest['portfolio_metrics']['Total Return']*100:.2f}%",
                    f"{backtest['portfolio_metrics']['Annual Return']*100:.2f}%",
                    f"{backtest['portfolio_metrics']['Annual Volatility']*100:.2f}%",
                    f"{backtest['portfolio_metrics']['Sharpe Ratio']:.2f}",
                    f"{backtest['portfolio_metrics']['Sortino Ratio']:.2f}",
                    f"{backtest['portfolio_metrics']['Max Drawdown']*100:.2f}%"
                ],
                "Equal Weight": [
                    f"{backtest['equal_weight_metrics']['Total Return']*100:.2f}%",
                    f"{backtest['equal_weight_metrics']['Annual Return']*100:.2f}%",
                    f"{backtest['equal_weight_metrics']['Annual Volatility']*100:.2f}%",
                    f"{backtest['equal_weight_metrics']['Sharpe Ratio']:.2f}",
                    f"{backtest['equal_weight_metrics']['Sortino Ratio']:.2f}",
                    f"{backtest['equal_weight_metrics']['Max Drawdown']*100:.2f}%"
                ]
            }

            if backtest["benchmark_metrics"]:
                metrics_data["SPY Benchmark"] = [
                    f"{backtest['benchmark_metrics']['Total Return']*100:.2f}%",
                    f"{backtest['benchmark_metrics']['Annual Return']*100:.2f}%",
                    f"{backtest['benchmark_metrics']['Annual Volatility']*100:.2f}%",
                    f"{backtest['benchmark_metrics']['Sharpe Ratio']:.2f}",
                    f"{backtest['benchmark_metrics']['Sortino Ratio']:.2f}",
                    f"{backtest['benchmark_metrics']['Max Drawdown']*100:.2f}%"
                ]

            st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

            # Drawdown chart
            col1, col2 = st.columns(2)
            with col1:
                fig = DashboardCharts.plot_drawdown(backtest["portfolio_cumulative"])
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Expected Performance (Training Data)")

                expected_df = pd.DataFrame([
                    {
                        "Portfolio": "Max Sharpe",
                        "Expected Return": f"{max_sharpe.expected_return*100:.2f}%",
                        "Expected Volatility": f"{max_sharpe.expected_volatility*100:.2f}%",
                        "Sharpe Ratio": f"{max_sharpe.sharpe_ratio:.2f}"
                    },
                    {
                        "Portfolio": "Min Volatility",
                        "Expected Return": f"{min_vol.expected_return*100:.2f}%",
                        "Expected Volatility": f"{min_vol.expected_volatility*100:.2f}%",
                        "Sharpe Ratio": f"{min_vol.sharpe_ratio:.2f}"
                    },
                    {
                        "Portfolio": "Equal Weight",
                        "Expected Return": f"{equal_weight.expected_return*100:.2f}%",
                        "Expected Volatility": f"{equal_weight.expected_volatility*100:.2f}%",
                        "Sharpe Ratio": f"{equal_weight.sharpe_ratio:.2f}"
                    }
                ])
                st.dataframe(expected_df, hide_index=True, use_container_width=True)

                # Asset statistics
                st.markdown("### Individual Asset Statistics")
                stats = st.session_state.loader.get_summary_statistics()
                stats = stats.sort_values(by="Sharpe Ratio", ascending=False)
                stats_display = stats.copy()
                stats_display["Annual Return"] = stats_display["Annual Return"].apply(
                    lambda x: f"{x*100:.2f}%"
                )
                stats_display["Annual Volatility"] = stats_display["Annual Volatility"].apply(
                    lambda x: f"{x*100:.2f}%"
                )
                stats_display["Sharpe Ratio"] = stats_display["Sharpe Ratio"].apply(
                    lambda x: f"{x:.2f}"
                )
                st.dataframe(stats_display, use_container_width=True)

            with col2:
                # Correlation heatmap
                fig = DashboardCharts.plot_correlation_heatmap(
                    st.session_state.train_returns
                )
                st.plotly_chart(fig, use_container_width=True)

            weights_col1, weights_col2 = st.columns(2)
            with weights_col1:
                fig = DashboardCharts.plot_weights_bar(
                    max_sharpe.weights,
                    title="Max Sharpe Portfolio Weights",
                    colorscale="Greens"
                )
                st.plotly_chart(fig, use_container_width=True)

            with weights_col2:
                fig = DashboardCharts.plot_weights_bar(
                    min_vol.weights,
                    title="Min Volatility Portfolio Weights",
                    colorscale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        # Initial state - show instructions
        st.info(
            "Configure your portfolio settings in the sidebar and click "
            "'Run Optimization' to begin."
        )

        st.markdown("""
        ### How to Use

        1. **Select Assets**: Choose stocks from the dropdown or add custom tickers
        2. **Set Date Range**: Define the historical period for analysis
        3. **Configure Split**: Set the train/test split date for backtesting
        4. **Adjust Risk-Free Rate**: Set the benchmark rate for Sharpe calculations
        5. **Run Optimization**: Click the button to find optimal portfolios

        ### What You'll See

        - **Efficient Frontier**: Visualization of risk-return tradeoffs
        - **Optimal Weights**: Recommended portfolio allocation
        - **Backtest Results**: Out-of-sample performance comparison
        - **Risk Metrics**: Sharpe, Sortino, Maximum Drawdown
        """)


if __name__ == "__main__":
    main()
