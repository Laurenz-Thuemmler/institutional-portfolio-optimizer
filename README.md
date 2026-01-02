# Portfolio Optimization Engine

An institutional-grade portfolio optimization dashboard built on Modern Portfolio Theory (MPT). This application provides interactive tools for constructing optimal investment portfolios, visualizing the efficient frontier, and backtesting strategies.

## Features

- **Institutional Universe**: Default universe includes Top 50 S&P 500 constituents
- **Diversification Controls**: User-defined maximum weight constraints to prevent over-concentration and force broad diversification
- **Efficient Frontier Visualization**: Interactive scatter plot of 2,000 random portfolios with the efficient frontier curve
- **Portfolio Optimization**: Find maximum Sharpe ratio and minimum volatility portfolios using scipy optimization
- **Backtesting Engine**: Out-of-sample performance testing against equal-weight and S&P 500 benchmarks
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown, and volatility calculations
- **Real-time Data**: Fetches historical prices from Yahoo Finance via yfinance

## Mathematical Background

### Modern Portfolio Theory

Modern Portfolio Theory (MPT), developed by Harry Markowitz in 1952, provides a mathematical framework for constructing portfolios that maximize expected return for a given level of risk.

#### Key Formulas

**Portfolio Return (Annualized)**
$$R_p = \sum_{i=1}^{n} w_i \cdot r_i \cdot 252$$

Where:
- $w_i$ = weight of asset $i$
- $r_i$ = mean daily return of asset $i$
- 252 = trading days per year

**Portfolio Volatility (Annualized)**
$$\sigma_p = \sqrt{w^T \Sigma w} \cdot \sqrt{252}$$

Where:
- $w$ = vector of portfolio weights
- $\Sigma$ = covariance matrix of daily returns

**Sharpe Ratio**
$$SR = \frac{R_p - R_f}{\sigma_p}$$

Where:
- $R_f$ = risk-free rate (annualized)

**Sortino Ratio**
$$SoR = \frac{R_p - R_f}{\sigma_{downside}}$$

Where downside deviation only considers negative returns.

**Maximum Drawdown**
$$MDD = \max_t \left( \frac{\text{Peak}_t - \text{Trough}_t}{\text{Peak}_t} \right)$$

### Efficient Frontier

The efficient frontier represents the set of optimal portfolios offering the highest expected return for each level of risk. The optimization problem is:

- **Maximize Sharpe**: $\max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$
- **Minimize Volatility**: $\min_w \sqrt{w^T \Sigma w}$

Subject to:
- $\sum w_i = 1$ (fully invested)
- $0 \leq w_i \leq w_{max}$ (long-only with diversification constraint)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Dashboard Guide

1. **Configure Assets**: Select stocks from the S&P 500 Top 50 or add custom tickers
2. **Set Date Range**: Choose the historical period for analysis (default: 3 years)
3. **Train/Test Split**: Define the split date - data before is used for optimization, after for backtesting
4. **Risk-Free Rate**: Adjust the benchmark rate for Sharpe calculations
5. **Max Allocation**: Set maximum weight per asset to force diversification (e.g., 5% = minimum 20 stocks)
6. **Run Optimization**: Click the button to execute the optimization

### Tabs

- **Optimization**: View the efficient frontier, optimal weights, and allocation chart
- **Backtest**: Compare optimized portfolio vs. equal-weight vs. SPY benchmark
- **Metrics**: Detailed performance statistics and correlation analysis

## Project Structure

```
portfolio-optimizer/
├── config.py               # Central configuration (tickers, dates, parameters)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore patterns
├── src/
│   ├── __init__.py         # Package initialization
│   ├── data_loader.py      # FinancialDataLoader class
│   ├── mathematics.py      # QuantMetrics class
│   ├── optimizer.py        # PortfolioOptimizer class
│   └── visualizer.py       # DashboardCharts class
├── tests/
│   └── test_mathematics.py # Unit tests for math functions
└── app.py                  # Streamlit dashboard entry point
```

## Running Tests

```bash
pytest tests/test_mathematics.py -v
```

Tests verify:
- Portfolio return calculations
- Portfolio volatility calculations
- Sharpe and Sortino ratio computations
- Maximum drawdown detection
- Covariance matrix regularization

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_TICKERS` | 49 stocks | S&P 500 Top 50 constituents |
| `BENCHMARK_TICKER` | SPY | Benchmark ETF |
| `RISK_FREE_RATE` | 0.05 | Annual risk-free rate (5%) |
| `TRADING_DAYS_PER_YEAR` | 252 | Trading days for annualization |
| `NUM_RANDOM_PORTFOLIOS` | 2000 | Portfolios for visualization |

## Technical Details

### Optimization

- **Method**: Sequential Least Squares Programming (SLSQP)
- **Constraints**: Weights sum to 1 (fully invested)
- **Bounds**: Long-only with max weight constraint (0 to max_weight for each asset)
- **Diversification**: User-configurable maximum allocation per asset prevents corner solutions
- **Fallback**: Equal weights if optimization fails

### Data Handling

- **Source**: Yahoo Finance via yfinance
- **Returns**: Log returns for statistical accuracy
- **Cleaning**: Forward-fill gaps, remove incomplete tickers
- **Validation**: Minimum 95% data completeness required

### Error Handling

- Graceful degradation when tickers fail to load
- Fallback to equal weights on optimization failure
- Covariance matrix regularization for numerical stability

## Dependencies

- `yfinance>=0.2.36` - Financial data
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.11.0` - Optimization
- `plotly>=5.18.0` - Interactive charts
- `streamlit>=1.31.0` - Dashboard framework
- `pytest>=7.4.0` - Testing

## Limitations

- **Historical Data Only**: Past performance does not guarantee future results
- **No Transaction Costs**: Real implementation would include fees and slippage
- **No Rebalancing**: Assumes buy-and-hold after optimization
- **Long-Only**: Short selling is not supported

## Future Enhancements

- [ ] Add transaction cost modeling
- [ ] Implement periodic rebalancing
- [ ] Support for short selling
- [ ] Monte Carlo simulation
- [ ] Black-Litterman model integration
- [ ] Risk parity optimization
- [ ] Export functionality (PDF reports)

## License

MIT License - See LICENSE file for details.

## References

- Markowitz, H. (1952). Portfolio Selection. The Journal of Finance.
- Sharpe, W. F. (1966). Mutual Fund Performance. The Journal of Business.
- Sortino, F. A., & Price, L. N. (1994). Performance Measurement in a Downside Risk Framework.
# institutional-portfolio-optimizer
