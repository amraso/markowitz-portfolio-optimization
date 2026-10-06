import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.optimize import minimize


# ============================================================
# CONFIGURATION
# ============================================================

TICKERS = ["SPY", "QQQ", "TLT", "GLD"]

START_DATE = "2015-01-01"

TRADING_DAYS = 252

N_PORTFOLIOS = 10_000

RANDOM_SEED = 42

FIGURE_DIR = "figures"


# ============================================================
# 1. DOWNLOAD HISTORICAL DATA
# ============================================================

os.makedirs(FIGURE_DIR, exist_ok=True)

data = yf.download(
    TICKERS,
    start=START_DATE,
    auto_adjust=True,
    progress=False,
)["Close"]

print("\nDownloaded price data:")
print(data.head())

print("\nAsset order:")
print(data.columns)


# ============================================================
# 2. DAILY LOG RETURNS
# ============================================================

log_returns = np.log(
    data / data.shift(1)
).dropna()

print("\nDaily log returns:")
print(log_returns.head())


# ============================================================
# 3. COVARIANCE AND CORRELATION
# ============================================================

covariance = log_returns.cov()
correlation = log_returns.corr()

print("\nCovariance matrix:")
print(covariance)

print("\nCorrelation matrix:")
print(correlation)


# ============================================================
# 4. NORMALIZED ASSET PERFORMANCE
# ============================================================

normalized_prices = (
    data / data.iloc[0] * 100
)

plt.figure(figsize=(10, 6))

for ticker in normalized_prices.columns:
    plt.plot(
        normalized_prices.index,
        normalized_prices[ticker],
        label=ticker
    )

plt.xlabel("Date")
plt.ylabel("Normalized price")
plt.title("Asset Performance — Starting Value = 100")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/normalized_prices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 5. CORRELATION MATRIX
# ============================================================

labels = correlation.columns

plt.figure(figsize=(8, 6))

plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(labels)),
    labels
)

plt.yticks(
    range(len(labels)),
    labels
)

for i in range(len(labels)):
    for j in range(len(labels)):

        value = correlation.iloc[i, j]

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center"
        )

plt.title("Asset Return Correlation Matrix")

plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 6. INDIVIDUAL ASSET STATISTICS
# ============================================================

mean_returns = log_returns.mean()

asset_returns_annual = (
    mean_returns * TRADING_DAYS
)

asset_volatility_annual = (
    log_returns.std()
    * np.sqrt(TRADING_DAYS)
)

print("\nIndividual asset statistics:")

for ticker in data.columns:

    print(
        f"{ticker}: "
        f"return = {asset_returns_annual[ticker]:.2%}, "
        f"volatility = {asset_volatility_annual[ticker]:.2%}"
    )


# ============================================================
# 7. EQUAL-WEIGHT PORTFOLIO
# ============================================================

weights = pd.Series(
    1 / len(data.columns),
    index=data.columns
)

print("\nEqual-weight portfolio:")

for ticker, weight in weights.items():
    print(f"{ticker}: {weight:.2%}")

print(
    f"\nSum of weights: "
    f"{weights.sum():.2f}"
)


# Expected return

portfolio_mean_daily = (
    weights @ mean_returns
)

portfolio_mean_annual = (
    portfolio_mean_daily
    * TRADING_DAYS
)


# Portfolio variance

portfolio_variance_daily = (
    weights
    @ covariance
    @ weights
)

portfolio_volatility_daily = np.sqrt(
    portfolio_variance_daily
)

portfolio_volatility_annual = (
    portfolio_volatility_daily
    * np.sqrt(TRADING_DAYS)
)


# Weighted average standalone volatility

weighted_average_volatility = (
    weights
    @ asset_volatility_annual
)


print("\nPortfolio statistics:")

print(
    f"Annualized expected return: "
    f"{portfolio_mean_annual:.2%}"
)

print(
    f"Annualized volatility: "
    f"{portfolio_volatility_annual:.2%}"
)

print(
    f"Weighted average individual volatility: "
    f"{weighted_average_volatility:.2%}"
)


# ============================================================
# 8. MONTE CARLO PORTFOLIO SIMULATION
# ============================================================

portfolio_returns = []
portfolio_volatilities = []
portfolio_weights = []

rng = np.random.default_rng(
    RANDOM_SEED
)

for _ in range(N_PORTFOLIOS):

    random_weights = rng.random(
        len(data.columns)
    )

    random_weights /= (
        random_weights.sum()
    )

    portfolio_return = (
        random_weights
        @ mean_returns
        * TRADING_DAYS
    )

    portfolio_variance = (
        random_weights
        @ covariance
        @ random_weights
    )

    portfolio_volatility = (
        np.sqrt(portfolio_variance)
        * np.sqrt(TRADING_DAYS)
    )

    portfolio_returns.append(
        portfolio_return
    )

    portfolio_volatilities.append(
        portfolio_volatility
    )

    portfolio_weights.append(
        random_weights
    )


portfolio_returns = np.array(
    portfolio_returns
)

portfolio_volatilities = np.array(
    portfolio_volatilities
)

portfolio_weights = np.array(
    portfolio_weights
)


# ============================================================
# 9. MONTE CARLO MINIMUM-VOLATILITY PORTFOLIO
# ============================================================

min_vol_index = np.argmin(
    portfolio_volatilities
)

min_volatility = (
    portfolio_volatilities[
        min_vol_index
    ]
)

min_vol_return = (
    portfolio_returns[
        min_vol_index
    ]
)

min_vol_weights = (
    portfolio_weights[
        min_vol_index
    ]
)


print(
    "\nMonte Carlo minimum-volatility portfolio:"
)

print(
    f"Expected return: "
    f"{min_vol_return:.2%}"
)

print(
    f"Volatility: "
    f"{min_volatility:.2%}"
)

print("\nWeights:")

for ticker, weight in zip(
    data.columns,
    min_vol_weights
):
    print(
        f"{ticker}: "
        f"{weight:.2%}"
    )


# ============================================================
# 10. RANDOM PORTFOLIO PLOT
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    portfolio_volatilities,
    portfolio_returns,
    s=10,
    alpha=0.4
)

plt.scatter(
    min_volatility,
    min_vol_return,
    s=150,
    marker="*",
    label="Monte Carlo minimum"
)

plt.xlabel(
    "Annualized Volatility"
)

plt.ylabel(
    "Annualized Expected Return"
)

plt.title(
    "Random Portfolios: Risk vs Return"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/random_portfolios.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. EXACT MINIMUM-VARIANCE OPTIMIZATION
# ============================================================

n_assets = len(data.columns)

initial_weights = (
    np.ones(n_assets)
    / n_assets
)


def portfolio_variance(
    weights,
    covariance_matrix
):
    """
    Annualized portfolio variance.
    """

    return (
        weights
        @ covariance_matrix
        @ weights
        * TRADING_DAYS
    )


def portfolio_variance_gradient(
    weights,
    covariance_matrix
):
    """
    Analytical gradient of annualized
    portfolio variance.
    """

    return (
        2
        * covariance_matrix.to_numpy()
        @ weights
        * TRADING_DAYS
    )


constraints = (
    {
        "type": "eq",
        "fun": lambda weights:
            np.sum(weights) - 1
    },
)


bounds = tuple(
    (0, 1)
    for _ in range(n_assets)
)


result = minimize(
    portfolio_variance,
    initial_weights,
    args=(covariance,),
    jac=portfolio_variance_gradient,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        "ftol": 1e-12,
        "maxiter": 1000
    }
)


optimal_weights = result.x


optimal_return = (
    optimal_weights
    @ mean_returns
    * TRADING_DAYS
)


optimal_volatility = np.sqrt(
    optimal_weights
    @ covariance
    @ optimal_weights
    * TRADING_DAYS
)


# ============================================================
# 12. OPTIMIZATION RESULTS
# ============================================================

print(
    "\nExact minimum-volatility portfolio:"
)

for ticker, weight in zip(
    data.columns,
    optimal_weights
):
    print(
        f"{ticker}: "
        f"{weight:.2%}"
    )


print(
    f"\nExpected annual return: "
    f"{optimal_return:.2%}"
)

print(
    f"Annualized volatility: "
    f"{optimal_volatility:.2%}"
)

print(
    "\nOptimizer success:",
    result.success
)

print(
    "Optimizer message:",
    result.message
)
