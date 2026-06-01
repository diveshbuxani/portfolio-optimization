# Portfolio Optimization (Max Sharpe)

I put this together as a clean, minimal example of maximizing the Sharpe ratio with real market data. It pulls daily close prices from Yahoo Finance and solves for a long-only portfolio. There is a notebook for exploration and a matching script for easy viewing on GitHub.

## What it does

- Downloads historical prices for a set of tickers.
- Computes annualized returns and covariance.
- Solves for the maximum Sharpe ratio portfolio under long-only constraints.
- Plots a Monte Carlo efficient frontier and highlights the optimal portfolio.

## Requirements

- Python 3.9+ (any recent 3.x should work)
- Packages: numpy, pandas, yfinance, scipy, matplotlib

## Setup

```bash
pip install numpy pandas yfinance scipy matplotlib
```

## Usage

### Run the script

```bash
python optimization.py
```

### Run the notebook

Open [optimization.ipynb](optimization.ipynb) in Jupyter or VS Code and run all cells.

## Notes

- Update the `tickers`, date range, or `risk_free_rate` in [optimization.py](optimization.py) to fit your use case.
- The script uses daily data and annualizes using 252 trading days.
- If you get empty data back, double-check ticker symbols and date ranges.

## Output

- Printed optimal weights and Sharpe ratio.
- A chart of the efficient frontier with the max-Sharpe portfolio highlighted.
