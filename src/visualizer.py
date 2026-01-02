"""
Dashboard Visualization Module.

This module provides interactive Plotly charts for the portfolio optimization
dashboard. All charts are designed for professional presentation and include
proper formatting, colors, and interactivity.

Charts Included:
    - Efficient Frontier with random portfolios
    - Backtest performance comparison
    - Asset allocation donut chart
    - Correlation heatmap
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Professional color palette
COLORS = {
    "primary": "#1f77b4",      # Blue
    "secondary": "#ff7f0e",    # Orange
    "success": "#2ca02c",      # Green
    "danger": "#d62728",       # Red
    "purple": "#9467bd",       # Purple
    "teal": "#17becf",         # Teal
    "gray": "#7f7f7f",         # Gray
    "background": "#fafafa",   # Light gray background
}


class DashboardCharts:
    """
    Creates interactive Plotly charts for portfolio visualization.

    All methods are static and return Plotly figure objects that can be
    displayed directly in Streamlit using st.plotly_chart().

    Example:
        >>> fig = DashboardCharts.plot_efficient_frontier(
        ...     random_portfolios, frontier_points, max_sharpe, min_vol
        ... )
        >>> st.plotly_chart(fig, use_container_width=True)
    """

    @staticmethod
    def plot_efficient_frontier(
        random_portfolios: pd.DataFrame,
        frontier_points: pd.DataFrame,
        max_sharpe_result: Dict,
        min_vol_result: Dict,
        height: int = 600
    ) -> go.Figure:
        """
        Create an interactive Efficient Frontier visualization.

        Displays:
        - Scatter plot of random portfolios colored by Sharpe ratio
        - Efficient frontier curve
        - Maximum Sharpe ratio portfolio (star marker)
        - Minimum volatility portfolio (diamond marker)

        Args:
            random_portfolios: DataFrame with Return, Volatility, Sharpe columns.
            frontier_points: DataFrame with efficient frontier coordinates.
            max_sharpe_result: Dict with expected_return, expected_volatility.
            min_vol_result: Dict with expected_return, expected_volatility.
            height: Chart height in pixels.

        Returns:
            Plotly Figure object.
        """
        fig = go.Figure()

        # Random portfolios scatter (colored by Sharpe ratio)
        fig.add_trace(
            go.Scatter(
                x=random_portfolios["Volatility"] * 100,
                y=random_portfolios["Return"] * 100,
                mode="markers",
                marker=dict(
                    size=6,
                    color=random_portfolios["Sharpe"],
                    colorscale="Viridis",
                    colorbar=dict(
                        title=dict(text="Sharpe Ratio", side="right")
                    ),
                    opacity=0.6,
                    line=dict(width=0)
                ),
                text=[
                    f"Return: {r:.2f}%<br>Vol: {v:.2f}%<br>Sharpe: {s:.2f}"
                    for r, v, s in zip(
                        random_portfolios["Return"] * 100,
                        random_portfolios["Volatility"] * 100,
                        random_portfolios["Sharpe"]
                    )
                ],
                hoverinfo="text",
                name="Random Portfolios"
            )
        )

        # Efficient frontier line
        if not frontier_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=frontier_points["Volatility"] * 100,
                    y=frontier_points["Return"] * 100,
                    mode="lines",
                    line=dict(color=COLORS["danger"], width=3),
                    name="Efficient Frontier"
                )
            )

        # Maximum Sharpe ratio portfolio
        fig.add_trace(
            go.Scatter(
                x=[max_sharpe_result["expected_volatility"] * 100],
                y=[max_sharpe_result["expected_return"] * 100],
                mode="markers",
                marker=dict(
                    size=20,
                    color=COLORS["success"],
                    symbol="star",
                    line=dict(color="white", width=2)
                ),
                name=f"Max Sharpe (SR: {max_sharpe_result['sharpe_ratio']:.2f})",
                hovertext=f"Max Sharpe Portfolio<br>"
                         f"Return: {max_sharpe_result['expected_return']*100:.2f}%<br>"
                         f"Volatility: {max_sharpe_result['expected_volatility']*100:.2f}%<br>"
                         f"Sharpe: {max_sharpe_result['sharpe_ratio']:.2f}",
                hoverinfo="text"
            )
        )

        # Minimum volatility portfolio
        fig.add_trace(
            go.Scatter(
                x=[min_vol_result["expected_volatility"] * 100],
                y=[min_vol_result["expected_return"] * 100],
                mode="markers",
                marker=dict(
                    size=18,
                    color=COLORS["primary"],
                    symbol="diamond",
                    line=dict(color="white", width=2)
                ),
                name=f"Min Volatility (Vol: {min_vol_result['expected_volatility']*100:.1f}%)",
                hovertext=f"Minimum Volatility Portfolio<br>"
                         f"Return: {min_vol_result['expected_return']*100:.2f}%<br>"
                         f"Volatility: {min_vol_result['expected_volatility']*100:.2f}%<br>"
                         f"Sharpe: {min_vol_result['sharpe_ratio']:.2f}",
                hoverinfo="text"
            )
        )

        fig.update_layout(
            title=dict(
                text="Efficient Frontier - Portfolio Optimization",
                font=dict(size=20)
            ),
            xaxis=dict(
                title="Annual Volatility (%)",
                tickformat=".1f",
                gridcolor="lightgray"
            ),
            yaxis=dict(
                title="Annual Return (%)",
                tickformat=".1f",
                gridcolor="lightgray"
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,1)",
                font=dict(color="black")
            ),
            height=height,
            template="plotly_white",
            hovermode="closest"
        )

        return fig

    @staticmethod
    def plot_backtest_performance(
        optimized_cumulative: pd.Series,
        equal_weight_cumulative: pd.Series,
        benchmark_cumulative: Optional[pd.Series] = None,
        height: int = 500
    ) -> go.Figure:
        """
        Create a cumulative returns comparison chart.

        Compares the performance of:
        - Optimized portfolio (based on training data)
        - Equal-weighted portfolio
        - Benchmark (SPY)

        Args:
            optimized_cumulative: Cumulative returns of optimized portfolio.
            equal_weight_cumulative: Cumulative returns of equal-weight portfolio.
            benchmark_cumulative: Cumulative returns of benchmark (optional).
            height: Chart height in pixels.

        Returns:
            Plotly Figure object.
        """
        fig = go.Figure()

        # Optimized portfolio
        fig.add_trace(
            go.Scatter(
                x=optimized_cumulative.index,
                y=(optimized_cumulative - 1) * 100,
                mode="lines",
                name="Optimized Portfolio",
                line=dict(color=COLORS["success"], width=2.5),
                hovertemplate="Date: %{x}<br>Return: %{y:.2f}%<extra></extra>"
            )
        )

        # Equal weight portfolio
        fig.add_trace(
            go.Scatter(
                x=equal_weight_cumulative.index,
                y=(equal_weight_cumulative - 1) * 100,
                mode="lines",
                name="Equal Weight",
                line=dict(color=COLORS["secondary"], width=2.5, dash="dash"),
                hovertemplate="Date: %{x}<br>Return: %{y:.2f}%<extra></extra>"
            )
        )

        # Benchmark (if provided)
        if benchmark_cumulative is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=(benchmark_cumulative - 1) * 100,
                    mode="lines",
                    name="SPY Benchmark",
                    line=dict(color=COLORS["gray"], width=2, dash="dot"),
                    hovertemplate="Date: %{x}<br>Return: %{y:.2f}%<extra></extra>"
                )
            )

        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

        fig.update_layout(
            title=dict(
                text="Out-of-Sample Backtest Performance",
                font=dict(size=20)
            ),
            xaxis=dict(
                title="Date",
                gridcolor="lightgray"
            ),
            yaxis=dict(
                title="Cumulative Return (%)",
                tickformat=".1f",
                gridcolor="lightgray"
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                font=dict(color="black")
            ),
            height=height,
            template="plotly_white",
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def plot_allocation_donut(
        weights: Dict[str, float],
        height: int = 450
    ) -> go.Figure:
        """
        Create a donut chart showing portfolio allocation.

        Args:
            weights: Dictionary of {ticker: weight}.
            height: Chart height in pixels.

        Returns:
            Plotly Figure object.
        """
        # Filter out zero/negligible weights
        filtered_weights = {
            k: v for k, v in weights.items() if v > 0.001
        }

        labels = list(filtered_weights.keys())
        values = list(filtered_weights.values())

        # Sort by weight descending
        sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_pairs) if sorted_pairs else ([], [])

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    textinfo="label+percent",
                    textposition="outside",
                    marker=dict(
                        colors=px.colors.qualitative.Set2[:len(labels)],
                        line=dict(color="white", width=2)
                    ),
                    hovertemplate="<b>%{label}</b><br>"
                                 "Weight: %{value:.2%}<br>"
                                 "<extra></extra>"
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text="Optimal Portfolio Allocation",
                font=dict(size=20)
            ),
            height=height,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(color="black")
            ),
            annotations=[
                dict(
                    text="Weights",
                    x=0.5,
                    y=0.5,
                    font_size=16,
                    showarrow=False
                )
            ]
        )

        return fig

    @staticmethod
    def plot_correlation_heatmap(
        returns: pd.DataFrame,
        height: int = 500
    ) -> go.Figure:
        """
        Create a correlation matrix heatmap.

        Args:
            returns: DataFrame of asset returns.
            height: Chart height in pixels.

        Returns:
            Plotly Figure object.
        """
        corr_matrix = returns.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>"
                             "Correlation: %{z:.3f}<extra></extra>"
            )
        )

        fig.update_layout(
            title=dict(
                text="Asset Correlation Matrix",
                font=dict(size=20)
            ),
            height=height,
            xaxis=dict(tickangle=45),
            template="plotly_white"
        )

        return fig

    @staticmethod
    def plot_weights_bar(
        weights: Dict[str, float],
        height: int = 400,
        title: str = "Portfolio Weights",
        colorscale: str = "Blues",
        min_weight: float = 0.001
    ) -> go.Figure:
        """
        Create a horizontal bar chart of portfolio weights.

        Args:
            weights: Dictionary of {ticker: weight}.
            height: Chart height in pixels.
            title: Chart title.
            colorscale: Plotly colorscale name for bar colors.
            min_weight: Minimum weight threshold to include.

        Returns:
            Plotly Figure object.
        """
        filtered_weights = {
            k: v for k, v in weights.items() if v > min_weight
        }
        if not filtered_weights:
            filtered_weights = weights

        # Sort by weight
        sorted_items = sorted(
            filtered_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        tickers = [item[0] for item in sorted_items]
        weight_values = [item[1] * 100 for item in sorted_items]
        max_weight = max(weight_values) if weight_values else 0

        fig = go.Figure(
            data=[
                go.Bar(
                    x=weight_values,
                    y=tickers,
                    orientation="h",
                    marker=dict(
                        color=weight_values,
                        colorscale=colorscale,
                        line=dict(color="white", width=1)
                    ),
                    text=[f"{w:.1f}%" for w in weight_values],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Weight: %{x:.2f}%<extra></extra>"
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20)
            ),
            xaxis=dict(
                title="Weight (%)",
                range=[0, max_weight * 1.15] if max_weight else None
            ),
            yaxis=dict(
                title="",
                autorange="reversed"
            ),
            height=height,
            template="plotly_white"
        )

        return fig

    @staticmethod
    def plot_drawdown(
        cumulative_returns: pd.Series,
        height: int = 350
    ) -> go.Figure:
        """
        Create a drawdown chart.

        Args:
            cumulative_returns: Series of cumulative returns.
            height: Chart height in pixels.

        Returns:
            Plotly Figure object.
        """
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max * 100

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill="tozeroy",
                fillcolor="rgba(214, 39, 40, 0.3)",
                line=dict(color=COLORS["danger"], width=1),
                name="Drawdown",
                hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>"
            )
        )

        fig.update_layout(
            title=dict(
                text="Portfolio Drawdown",
                font=dict(size=18)
            ),
            xaxis=dict(title="Date", gridcolor="lightgray"),
            yaxis=dict(
                title="Drawdown (%)",
                tickformat=".1f",
                gridcolor="lightgray"
            ),
            height=height,
            template="plotly_white"
        )

        return fig

    @staticmethod
    def create_performance_table(
        metrics: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """
        Create a performance comparison table.

        Args:
            metrics: Nested dict of {strategy: {metric_name: value}}.

        Returns:
            Plotly Figure object with table.
        """
        strategies = list(metrics.keys())
        metric_names = list(metrics[strategies[0]].keys()) if strategies else []

        # Build table data
        header_values = ["Metric"] + strategies
        cell_values = [[name for name in metric_names]]

        for strategy in strategies:
            values = []
            for metric_name in metric_names:
                value = metrics[strategy].get(metric_name, 0)
                if "Return" in metric_name or "Volatility" in metric_name or "Drawdown" in metric_name:
                    values.append(f"{value*100:.2f}%")
                else:
                    values.append(f"{value:.2f}")
            cell_values.append(values)

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=header_values,
                        fill_color=COLORS["primary"],
                        font=dict(color="white", size=14),
                        align="center",
                        height=35
                    ),
                    cells=dict(
                        values=cell_values,
                        fill_color=[
                            ["white", "#f5f5f5"] * (len(metric_names) // 2 + 1)
                        ],
                        font=dict(size=13),
                        align="center",
                        height=30
                    )
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text="Performance Comparison",
                font=dict(size=18)
            ),
            height=len(metric_names) * 40 + 100
        )

        return fig
