"""Financial analysis and risk visualization for the TSX portfolio project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_preprocessing import load_raw_data


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "Portfolio Project.csv"
VISUALS_DIR = PROJECT_DIR / "visuals"
REPORTS_DIR = PROJECT_DIR / "reports"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.035


def prepare_returns() -> pd.DataFrame:
    """Load the raw table and add return/risk fields."""
    df = load_raw_data(DATA_PATH)
    df = df.dropna(subset=["daily_return", "close_price", "volume"]).copy()

    market = (
        df.drop_duplicates("trade_date")
        .sort_values("trade_date")
        .set_index("trade_date")["tsx_total_return_index"]
        .pct_change()
        .rename("market_return")
    )
    df["market_return"] = df["trade_date"].map(market)

    df["dollar_volume"] = df["close_price"] * df["volume"]
    df["return_index"] = df.groupby("ticker")["daily_return"].transform(lambda x: (1 + x).cumprod() * 100)
    df["running_max_index"] = df.groupby("ticker")["return_index"].cummax()
    df["drawdown"] = df["return_index"] / df["running_max_index"] - 1
    df["rolling_volatility_60d"] = (
        df.groupby("ticker")["daily_return"]
        .rolling(60)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(TRADING_DAYS)
    )
    return df


def calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate investment and downside-risk metrics by ticker."""
    market_returns = df.drop_duplicates("trade_date").set_index("trade_date")["market_return"].dropna()
    market_cumulative_return = (1 + market_returns).prod() - 1
    market_annualized_return = (1 + market_cumulative_return) ** (TRADING_DAYS / len(market_returns)) - 1
    records = []

    for ticker, group in df.groupby("ticker"):
        returns = group.set_index("trade_date")["daily_return"].dropna()
        aligned = pd.concat(
            [returns.rename("stock_return"), market_returns.rename("market_return")],
            axis=1,
            sort=False,
        ).dropna()

        cumulative_return = (1 + returns).prod() - 1
        annualized_return = (1 + cumulative_return) ** (TRADING_DAYS / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS)
        excess_annual_return = annualized_return - RISK_FREE_RATE
        sharpe_ratio = excess_annual_return / annualized_volatility if annualized_volatility else np.nan
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(TRADING_DAYS)
        sortino_ratio = excess_annual_return / downside_deviation if downside_deviation else np.nan
        max_drawdown = group["drawdown"].min()
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        beta = aligned["stock_return"].cov(aligned["market_return"]) / aligned["market_return"].var()
        capm_expected_return = RISK_FREE_RATE + beta * (market_annualized_return - RISK_FREE_RATE)
        jensens_alpha = annualized_return - capm_expected_return
        treynor_ratio = excess_annual_return / beta if beta else np.nan
        active_return = aligned["stock_return"] - aligned["market_return"]
        tracking_error = active_return.std() * np.sqrt(TRADING_DAYS)
        information_ratio = (annualized_return - market_annualized_return) / tracking_error if tracking_error else np.nan

        records.append(
            {
                "ticker": ticker,
                "company_name": group["company_name"].iloc[0],
                "business": group["business"].iloc[0],
                "observations": len(returns),
                "cumulative_return": cumulative_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "var_95_daily": var_95,
                "cvar_95_daily": cvar_95,
                "beta_to_tsx": beta,
                "capm_expected_return": capm_expected_return,
                "jensens_alpha": jensens_alpha,
                "treynor_ratio": treynor_ratio,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "positive_day_rate": returns.gt(0).mean(),
                "average_dollar_volume": group["dollar_volume"].mean(),
            }
        )

    metrics = pd.DataFrame(records)
    return metrics.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)


def save_visuals(df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Create portfolio-ready financial analysis charts."""
    VISUALS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    palette = sns.color_palette("tab20", n_colors=df["ticker"].nunique())

    plt.figure(figsize=(13, 7))
    sns.lineplot(data=df, x="trade_date", y="return_index", hue="ticker", palette=palette, linewidth=1.8)
    plt.axhline(100, color="black", linewidth=0.8, alpha=0.5)
    plt.title("Indexed Total Return by Stock, Base = 100")
    plt.xlabel("Trade date")
    plt.ylabel("Growth of $100 investment")
    plt.legend(title="Ticker", ncol=3, frameon=True)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "indexed_total_return.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 7))
    scatter = sns.scatterplot(
        data=metrics,
        x="annualized_volatility",
        y="annualized_return",
        size=metrics["max_drawdown"].abs(),
        hue="beta_to_tsx",
        sizes=(80, 450),
        palette="coolwarm",
        edgecolor="black",
        linewidth=0.7,
    )
    for _, row in metrics.iterrows():
        scatter.text(row["annualized_volatility"] + 0.002, row["annualized_return"], row["ticker"], fontsize=9)
    plt.title("Risk-Return Profile by Stock")
    plt.xlabel("Annualized volatility")
    plt.ylabel("Annualized return")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "risk_return_scatter.png", dpi=180)
    plt.close()

    pivot_returns = df.pivot_table(index="trade_date", columns="ticker", values="daily_return", aggfunc="mean")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_returns.corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Daily Return Correlation Matrix")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "return_correlation_heatmap.png", dpi=180)
    plt.close()

    grid = sns.FacetGrid(df, col="ticker", col_wrap=4, height=2.2, sharey=True)
    grid.map_dataframe(sns.lineplot, x="trade_date", y="drawdown", color="#C44E52", linewidth=1.3)
    grid.set_axis_labels("", "Drawdown")
    grid.set_titles("{col_name}")
    for ax in grid.axes.flat:
        ax.axhline(0, color="black", linewidth=0.6)
        ax.tick_params(axis="x", labelrotation=45)
    grid.fig.suptitle("Drawdown by Stock", y=1.03)
    grid.tight_layout()
    grid.savefig(VISUALS_DIR / "drawdown_by_ticker.png", dpi=180)
    plt.close(grid.fig)

    plt.figure(figsize=(13, 7))
    sns.lineplot(data=df.dropna(subset=["rolling_volatility_60d"]), x="trade_date", y="rolling_volatility_60d", hue="ticker", palette=palette)
    plt.title("60-Day Rolling Annualized Volatility")
    plt.xlabel("Trade date")
    plt.ylabel("Annualized volatility")
    plt.legend(title="Ticker", ncol=3, frameon=True)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "rolling_volatility_60d.png", dpi=180)
    plt.close()

    tail = metrics.sort_values("var_95_daily")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tail, x="var_95_daily", y="ticker", color="#4C78A8", label="VaR 95%")
    sns.scatterplot(data=tail, x="cvar_95_daily", y="ticker", color="#C44E52", s=70, label="CVaR 95%")
    plt.title("Daily Downside Risk: Historical VaR and CVaR")
    plt.xlabel("Daily return threshold")
    plt.ylabel("Ticker")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "var_cvar_by_ticker.png", dpi=180)
    plt.close()

    alpha = metrics.sort_values("jensens_alpha", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=alpha, x="jensens_alpha", y="ticker", color="#59A14F")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Jensen's Alpha vs CAPM Expected Return")
    plt.xlabel("Annualized Jensen's alpha")
    plt.ylabel("Ticker")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "jensens_alpha_by_ticker.png", dpi=180)
    plt.close()

    ratios = metrics.sort_values("sortino_ratio", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=ratios, x="sortino_ratio", y="ticker", color="#F28E2B")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Sortino Ratio by Stock")
    plt.xlabel("Sortino ratio")
    plt.ylabel("Ticker")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "sortino_ratio_by_ticker.png", dpi=180)
    plt.close()


def write_financial_report(metrics: pd.DataFrame) -> None:
    """Write a concise financial analyst/risk analyst style report."""
    REPORTS_DIR.mkdir(exist_ok=True)
    top_sharpe = metrics.iloc[0]
    highest_vol = metrics.sort_values("annualized_volatility", ascending=False).iloc[0]
    worst_drawdown = metrics.sort_values("max_drawdown").iloc[0]
    highest_beta = metrics.sort_values("beta_to_tsx", ascending=False).iloc[0]
    lowest_var = metrics.sort_values("var_95_daily").iloc[0]
    highest_alpha = metrics.sort_values("jensens_alpha", ascending=False).iloc[0]
    highest_sortino = metrics.sort_values("sortino_ratio", ascending=False).iloc[0]

    table = metrics[
        [
            "ticker",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "var_95_daily",
            "cvar_95_daily",
            "beta_to_tsx",
            "capm_expected_return",
            "jensens_alpha",
            "treynor_ratio",
            "information_ratio",
            "positive_day_rate",
        ]
    ].copy()
    for column in table.columns:
        if column != "ticker":
            table[column] = table[column].map(lambda x: f"{x:.4f}")

    report = f"""# Financial Risk Analysis Report

## Executive Summary

This project analyzes daily TSX stock data from 2021 to 2025 and turns a course dataset into a Python-based financial analyst/risk analyst portfolio project. The analysis combines investment performance metrics, downside-risk measures, data visualization, and machine learning models for next-day return prediction.

The strongest risk-adjusted performer in this sample is **{top_sharpe['ticker']}**, with a Sharpe ratio of **{top_sharpe['sharpe_ratio']:.2f}** after applying a **{RISK_FREE_RATE:.1%}** annual risk-free rate assumption. The highest Sortino ratio belongs to **{highest_sortino['ticker']}**, emphasizing stronger return per unit of downside volatility. The highest annualized volatility belongs to **{highest_vol['ticker']}** at **{highest_vol['annualized_volatility']:.2%}**. The deepest maximum drawdown belongs to **{worst_drawdown['ticker']}** at **{worst_drawdown['max_drawdown']:.2%}**. The highest market beta belongs to **{highest_beta['ticker']}**, which indicates stronger sensitivity to broad TSX market movements. The most positive Jensen's alpha belongs to **{highest_alpha['ticker']}**, meaning it outperformed its CAPM-implied expected return by the largest margin in this sample.

## Risk Methodology

- **Annualized return** estimates the compounded yearly return based on daily returns.
- **Annualized volatility** measures return dispersion scaled to a 252-trading-day year.
- **Sharpe ratio** compares excess annualized return to annualized volatility.
- **Sortino ratio** compares excess annualized return to downside volatility only.
- **Maximum drawdown** measures the largest peak-to-trough loss from the indexed return path.
- **Historical VaR 95%** estimates the daily return threshold where the worst 5% of outcomes begin.
- **Historical CVaR 95%** measures the average loss conditional on returns falling below VaR.
- **Beta to TSX** measures stock sensitivity relative to the S&P/TSX Composite Total Return Index.
- **CAPM expected return** estimates required return as risk-free rate plus beta times market risk premium.
- **Jensen's alpha** compares realized annualized return against the CAPM-implied expected return.
- **Treynor ratio** measures excess return per unit of systematic market risk.
- **Tracking error** and **information ratio** compare stock performance against the TSX benchmark.

## Financial Economics Formula Appendix

The analysis uses the following formulas, which are common in financial analyst and risk analyst work:

```text
Annualized Return = (1 + Cumulative Return)^(252 / Number of Trading Days) - 1
Annualized Volatility = Standard Deviation(Daily Returns) * sqrt(252)
Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Annualized Volatility
Sortino Ratio = (Annualized Return - Risk-Free Rate) / Downside Deviation
Beta = Covariance(Stock Return, Market Return) / Variance(Market Return)
CAPM Expected Return = Risk-Free Rate + Beta * (Market Return - Risk-Free Rate)
Jensen's Alpha = Annualized Return - CAPM Expected Return
Treynor Ratio = (Annualized Return - Risk-Free Rate) / Beta
Historical VaR 95% = 5th Percentile of Daily Returns
Historical CVaR 95% = Average Daily Return Conditional on Return <= VaR 95%
Tracking Error = Standard Deviation(Stock Return - Market Return) * sqrt(252)
Information Ratio = (Annualized Return - Benchmark Annualized Return) / Tracking Error
```

WACC is also an important corporate finance formula, but it cannot be calculated from this market-price dataset alone because it requires capital structure and financing inputs. In a banking analyst setting, WACC would be calculated as:

```text
WACC = (E / (D + E)) * Cost of Equity + (D / (D + E)) * Cost of Debt * (1 - Tax Rate)
```

Required WACC inputs:

- Market value of equity
- Market value of debt
- Pre-tax cost of debt
- Corporate tax rate
- Cost of equity, often estimated using CAPM

## Risk Metrics

{table.to_markdown(index=False)}

## Visualization Outputs

- `visuals/indexed_total_return.png`: compares growth of a $100 investment across tickers.
- `visuals/risk_return_scatter.png`: shows annualized return versus volatility, with beta and drawdown context.
- `visuals/return_correlation_heatmap.png`: shows cross-stock return correlation.
- `visuals/drawdown_by_ticker.png`: shows peak-to-trough losses through time.
- `visuals/rolling_volatility_60d.png`: tracks changing annualized volatility.
- `visuals/var_cvar_by_ticker.png`: compares downside tail risk using VaR and CVaR.
- `visuals/jensens_alpha_by_ticker.png`: compares realized return against CAPM-implied expected return.
- `visuals/sortino_ratio_by_ticker.png`: compares excess return against downside volatility.

## Machine Learning Connection

The machine learning section uses engineered financial features such as lagged returns, rolling returns, rolling volatility, market return, excess return, dividends, and volume. Linear Regression is used to predict next-day return, while Logistic Regression, Decision Tree, and Random Forest models classify whether the next-day return is positive.

The predictive results are intentionally interpreted conservatively. Short-term stock returns are noisy, so the main value of this project is not claiming a trading edge. The value is demonstrating a professional workflow: financial feature engineering, risk measurement, model training, model evaluation, and clear communication of limitations.

## Analyst Takeaway

For a financial analyst or risk analyst role, this project demonstrates the ability to translate raw market data into risk metrics, visual dashboards, and model-based insights. The most important portfolio story is the end-to-end process: cleaning market data, building interpretable risk measures, comparing securities, and using Python to support analytical decision-making.
"""

    (REPORTS_DIR / "financial_risk_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    df = prepare_returns()
    metrics = calculate_risk_metrics(df)
    REPORTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(REPORTS_DIR / "risk_metrics.csv", index=False)
    save_visuals(df, metrics)
    write_financial_report(metrics)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
