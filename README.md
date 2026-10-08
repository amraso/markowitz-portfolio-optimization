# Project 1 — Markowitz Portfolio Optimization with Python

This project explores portfolio risk and diversification using historical market data.

The analysis uses four ETFs:

- SPY — S&P 500
- QQQ — Nasdaq-100
- TLT — Long-term US Treasury bonds
- GLD — Gold

## Features

- Historical price download using yfinance
- Daily logarithmic return calculation
- Covariance and correlation analysis
- Correlation matrix visualization
- Portfolio expected return and volatility
- Diversification analysis
- Monte Carlo generation of random portfolios
- Minimum-volatility portfolio search
- Constrained portfolio optimization using SLSQP

## Methodology

Portfolio expected return:

$$
\mu_p = w^T \mu
$$

Portfolio variance:

$$
\sigma_p^2 = w^T \Sigma w
$$

where:

- \(w\) is the vector of portfolio weights
- \(\mu\) is the vector of expected asset returns
- \(\Sigma\) is the covariance matrix

The numerical optimization minimizes portfolio variance subject to:

$$
\sum_i w_i = 1
$$

and:

$$
0 \leq w_i \leq 1
$$

which corresponds to a fully invested portfolio without short selling.

## Example Results

Using historical data from 2015 onward:

- Equal-weight portfolio return: approximately 9.9%
- Equal-weight portfolio volatility: approximately 11.3%
- Minimum-volatility portfolio volatility: approximately 9.6%

The optimized portfolio allocates capital across equities, bonds and gold in order to reduce total portfolio risk through diversification.

## Technologies

- Python
- NumPy
- pandas
- Matplotlib
- SciPy
- yfinance
