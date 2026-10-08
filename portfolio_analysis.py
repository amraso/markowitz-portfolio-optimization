"""
Portfolio Analysis and Markowitz Optimization
==============================================

This script performs a quantitative portfolio analysis using historical
ETF market data.

Main features
-------------
1. Download historical ETF prices.
2. Calculate daily logarithmic returns.
3. Compute covariance and correlation matrices.
4. Compare normalized asset performance.
5. Analyze an equal-weight portfolio.
6. Generate random portfolios using Monte Carlo simulation.
7. Find the minimum-volatility portfolio via Monte Carlo.
8. Find the exact minimum-variance portfolio using SLSQP.
9. Construct the Markowitz minimum-variance frontier.
10. Find the maximum-Sharpe portfolio using Monte Carlo.
11. Find the exact maximum-Sharpe portfolio using SLSQP.
12. Plot the Capital Allocation Line.

Assets
------
SPY : S&P 500 ETF
QQQ : Nasdaq-100 ETF
TLT : Long-term U.S. Treasury bond ETF
GLD : Gold ETF

Notes
-----
- Short selling is not allowed.
- Portfolio weights must sum to 1.
- The risk-free rate is set manually in the configuration section.
- Historical mean returns are used as estimates of expected returns.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Teaching assumption.
# This can later be replaced with a realistic risk-free rate.
RISK_FREE_RATE = 0.00

FIGURE_DIR = Path("figures")


# ============================================================
# PORTFOLIO FUNCTIONS
# ============================================================

def annualized_portfolio_return(weights, mean_returns):
    """
    Calculate annualized expected portfolio return.
    """

    return (
        weights
        @ mean_returns
        * TRADING_DAYS
    )


def annualized_portfolio_variance(
    weights,
    covariance_matrix
):
    """
    Calculate annualized portfolio variance.
    """

    return (
        weights
        @ covariance_matrix
        @ weights
        * TRADING_DAYS
    )


def annualized_portfolio_volatility(
    weights,
    covariance_matrix
):
    """
    Calculate annualized portfolio volatility.
    """

    return np.sqrt(
        annualized_portfolio_variance(
            weights,
            covariance_matrix
        )
    )


def portfolio_variance_gradient(
    weights,
    covariance_matrix
):
    """
    Analytical gradient of annualized portfolio variance.
    """

    return (
        2
        * covariance_matrix.to_numpy()
        @ weights
        * TRADING_DAYS
    )


def negative_sharpe_ratio(
    weights,
    mean_returns,
    covariance_matrix,
    risk_free_rate
):
    """
    Return the negative Sharpe ratio.

    scipy.optimize.minimize() performs minimization.
    Therefore, minimizing -Sharpe is equivalent to
    maximizing the Sharpe ratio.
    """

    portfolio_return = annualized_portfolio_return(
        weights,
        mean_returns
    )

    portfolio_volatility = annualized_portfolio_volatility(
        weights,
        covariance_matrix
    )

    sharpe_ratio = (
        portfolio_return - risk_free_rate
    ) / portfolio_volatility

    return -sharpe_ratio


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 1. DOWNLOAD HISTORICAL DATA
    # ========================================================

    data = yf.download(
        TICKERS,
        start=START_DATE,
        auto_adjust=True,
        progress=False
    )["Close"]

    if data.empty:
        raise RuntimeError(
            "No market data were downloaded."
        )

    data = data.dropna()

    print("\nDownloaded price data:")
    print(data.head())

    print("\nAsset order:")
    print(data.columns)


    # ========================================================
    # 2. DAILY LOG RETURNS
    # ========================================================

    log_returns = np.log(
        data / data.shift(1)
    ).dropna()

    print("\nDaily log returns:")
    print(log_returns.head())


    # ========================================================
    # 3. COVARIANCE AND CORRELATION
    # ========================================================

    covariance = log_returns.cov()

    correlation = log_returns.corr()

    print("\nCovariance matrix:")
    print(covariance)

    print("\nCorrelation matrix:")
    print(correlation)


    # ========================================================
    # 4. NORMALIZED ASSET PERFORMANCE
    # ========================================================

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
    plt.title(
        "Asset Performance — Starting Value = 100"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "normalized_prices.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # 5. CORRELATION MATRIX
    # ========================================================

    labels = correlation.columns

    plt.figure(figsize=(8, 6))

    plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(
        label="Correlation"
    )

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

    plt.title(
        "Asset Return Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # 6. INDIVIDUAL ASSET STATISTICS
    # ========================================================

    mean_returns = log_returns.mean()

    asset_returns_annual = (
        mean_returns
        * TRADING_DAYS
    )

    asset_volatility_annual = (
        log_returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    print(
        "\nIndividual asset statistics:"
    )

    for ticker in data.columns:

        print(
            f"{ticker}: "
            f"return = "
            f"{asset_returns_annual[ticker]:.2%}, "
            f"volatility = "
            f"{asset_volatility_annual[ticker]:.2%}"
        )


    # ========================================================
    # 7. EQUAL-WEIGHT PORTFOLIO
    # ========================================================

    weights = pd.Series(
        1 / len(data.columns),
        index=data.columns
    )

    print(
        "\nEqual-weight portfolio:"
    )

    for ticker, weight in weights.items():

        print(
            f"{ticker}: {weight:.2%}"
        )

    print(
        f"\nSum of weights: "
        f"{weights.sum():.2f}"
    )

    portfolio_mean_annual = (
        annualized_portfolio_return(
            weights,
            mean_returns
        )
    )

    portfolio_volatility_annual = (
        annualized_portfolio_volatility(
            weights,
            covariance
        )
    )

    weighted_average_volatility = (
        weights
        @ asset_volatility_annual
    )

    print(
        "\nEqual-weight portfolio statistics:"
    )

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


    # ========================================================
    # 8. MONTE CARLO PORTFOLIO SIMULATION
    # ========================================================

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
            annualized_portfolio_return(
                random_weights,
                mean_returns
            )
        )

        portfolio_volatility = (
            annualized_portfolio_volatility(
                random_weights,
                covariance
            )
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


    # ========================================================
    # 9. MONTE CARLO MINIMUM-VOLATILITY PORTFOLIO
    # ========================================================

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


    # ========================================================
    # 10. EXACT MINIMUM-VARIANCE OPTIMIZATION
    # ========================================================

    n_assets = len(
        data.columns
    )

    initial_weights = (
        np.ones(n_assets)
        / n_assets
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

    min_variance_result = minimize(
        annualized_portfolio_variance,
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

    if not min_variance_result.success:
        raise RuntimeError(
            "Minimum-variance optimization failed: "
            + min_variance_result.message
        )

    optimal_weights = (
        min_variance_result.x
    )

    optimal_return = (
        annualized_portfolio_return(
            optimal_weights,
            mean_returns
        )
    )

    optimal_volatility = (
        annualized_portfolio_volatility(
            optimal_weights,
            covariance
        )
    )

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


    # ========================================================
    # 11. MARKOWITZ MINIMUM-VARIANCE FRONTIER
    # ========================================================

    target_returns = np.linspace(
        asset_returns_annual.min(),
        asset_returns_annual.max(),
        100
    )

    frontier_returns = []
    frontier_volatilities = []
    frontier_weights_list = []

    for target_return in target_returns:

        frontier_constraints = (
            {
                "type": "eq",
                "fun": lambda weights:
                    np.sum(weights) - 1
            },

            {
                "type": "eq",

                "fun": (
                    lambda weights,
                    target=target_return:
                    annualized_portfolio_return(
                        weights,
                        mean_returns
                    )
                    - target
                )
            }
        )

        frontier_result = minimize(
            annualized_portfolio_variance,
            initial_weights,
            args=(covariance,),
            jac=portfolio_variance_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=frontier_constraints,
            options={
                "ftol": 1e-12,
                "maxiter": 1000
            }
        )

        if frontier_result.success:

            frontier_weights = (
                frontier_result.x
            )

            frontier_volatility = (
                annualized_portfolio_volatility(
                    frontier_weights,
                    covariance
                )
            )

            frontier_returns.append(
                target_return
            )

            frontier_volatilities.append(
                frontier_volatility
            )

            frontier_weights_list.append(
                frontier_weights
            )

    frontier_returns = np.array(
        frontier_returns
    )

    frontier_volatilities = np.array(
        frontier_volatilities
    )

    frontier_weights_list = np.array(
        frontier_weights_list
    )


    # ========================================================
    # 12. MONTE CARLO MAXIMUM-SHARPE PORTFOLIO
    # ========================================================

    portfolio_sharpe_ratios = (
        portfolio_returns
        - RISK_FREE_RATE
    ) / portfolio_volatilities

    max_sharpe_index = np.argmax(
        portfolio_sharpe_ratios
    )

    max_sharpe_return = (
        portfolio_returns[
            max_sharpe_index
        ]
    )

    max_sharpe_volatility = (
        portfolio_volatilities[
            max_sharpe_index
        ]
    )

    max_sharpe_weights = (
        portfolio_weights[
            max_sharpe_index
        ]
    )

    max_sharpe_ratio = (
        portfolio_sharpe_ratios[
            max_sharpe_index
        ]
    )

    print(
        "\nMonte Carlo maximum-Sharpe portfolio:"
    )

    print(
        f"Expected return: "
        f"{max_sharpe_return:.2%}"
    )

    print(
        f"Volatility: "
        f"{max_sharpe_volatility:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{max_sharpe_ratio:.3f}"
    )

    print("\nWeights:")

    for ticker, weight in zip(
        data.columns,
        max_sharpe_weights
    ):

        print(
            f"{ticker}: "
            f"{weight:.2%}"
        )


    # ========================================================
    # 13. EXACT MAXIMUM-SHARPE PORTFOLIO
    # ========================================================

    max_sharpe_result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(
            mean_returns,
            covariance,
            RISK_FREE_RATE
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000
        }
    )

    if not max_sharpe_result.success:
        raise RuntimeError(
            "Maximum-Sharpe optimization failed: "
            + max_sharpe_result.message
        )

    optimal_sharpe_weights = (
        max_sharpe_result.x
    )

    optimal_sharpe_return = (
        annualized_portfolio_return(
            optimal_sharpe_weights,
            mean_returns
        )
    )

    optimal_sharpe_volatility = (
        annualized_portfolio_volatility(
            optimal_sharpe_weights,
            covariance
        )
    )

    optimal_sharpe_ratio = (
        optimal_sharpe_return
        - RISK_FREE_RATE
    ) / optimal_sharpe_volatility

    print(
        "\nExact maximum-Sharpe portfolio:"
    )

    for ticker, weight in zip(
        data.columns,
        optimal_sharpe_weights
    ):

        print(
            f"{ticker}: "
            f"{weight:.2%}"
        )

    print(
        f"\nExpected annual return: "
        f"{optimal_sharpe_return:.2%}"
    )

    print(
        f"Annualized volatility: "
        f"{optimal_sharpe_volatility:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{optimal_sharpe_ratio:.3f}"
    )


    # ========================================================
    # 14. CAPITAL ALLOCATION LINE
    # ========================================================

    cal_volatility = np.linspace(
        0,
        frontier_volatilities.max(),
        200
    )

    cal_return = (
        RISK_FREE_RATE
        + optimal_sharpe_ratio
        * cal_volatility
    )


    # ========================================================
    # 15. FINAL RISK-RETURN PLOT
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    # Monte Carlo portfolios
    plt.scatter(
        portfolio_volatilities,
        portfolio_returns,
        s=10,
        alpha=0.20,
        label="Random portfolios"
    )

    # Minimum-variance frontier
    plt.plot(
        frontier_volatilities,
        frontier_returns,
        linewidth=2.5,
        label="Minimum-variance frontier"
    )

    # Capital Allocation Line
    plt.plot(
        cal_volatility,
        cal_return,
        linestyle="--",
        linewidth=2,
        label="Capital Allocation Line"
    )

    # Exact minimum-volatility portfolio
    plt.scatter(
        optimal_volatility,
        optimal_return,
        s=180,
        marker="*",
        label="Minimum-volatility portfolio"
    )

    # Monte Carlo maximum-Sharpe portfolio
    plt.scatter(
        max_sharpe_volatility,
        max_sharpe_return,
        s=130,
        marker="X",
        label="Monte Carlo max-Sharpe"
    )

    # Exact maximum-Sharpe / tangency portfolio
    plt.scatter(
        optimal_sharpe_volatility,
        optimal_sharpe_return,
        s=180,
        marker="*",
        label="Tangency / exact max-Sharpe"
    )

    # Risk-free asset
    plt.scatter(
        0,
        RISK_FREE_RATE,
        s=100,
        marker="o",
        label="Risk-free asset"
    )

    plt.xlabel(
        "Annualized Volatility"
    )

    plt.ylabel(
        "Annualized Expected Return"
    )

    plt.title(
        "Markowitz Portfolio Optimization"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "markowitz_portfolio_optimization.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
