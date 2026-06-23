# Financial Risk Analysis Report

## Executive Summary

This project analyzes daily TSX stock data from 2021 to 2025 and turns a course dataset into a Python-based financial analyst/risk analyst portfolio project. The analysis combines investment performance metrics, downside-risk measures, data visualization, and machine learning models for next-day return prediction.

The strongest risk-adjusted performer in this sample is **RY**, with a Sharpe ratio of **1.27** after applying a **3.5%** annual risk-free rate assumption. The highest Sortino ratio belongs to **RY**, emphasizing stronger return per unit of downside volatility. The highest annualized volatility belongs to **SU** at **31.40%**. The deepest maximum drawdown belongs to **BCE** at **-49.86%**. The highest market beta belongs to **CNQ**, which indicates stronger sensitivity to broad TSX market movements. The most positive Jensen's alpha belongs to **CNQ**, meaning it outperformed its CAPM-implied expected return by the largest margin in this sample.

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

| ticker   |   annualized_return |   annualized_volatility |   sharpe_ratio |   sortino_ratio |   max_drawdown |   var_95_daily |   cvar_95_daily |   beta_to_tsx |   capm_expected_return |   jensens_alpha |   treynor_ratio |   information_ratio |   positive_day_rate |
|:---------|--------------------:|------------------------:|---------------:|----------------:|---------------:|---------------:|----------------:|--------------:|-----------------------:|----------------:|----------------:|--------------------:|--------------------:|
| RY       |              0.2209 |                  0.1461 |         1.2729 |          1.6976 |        -0.2119 |        -0.0143 |         -0.0214 |        0.796  |                 0.1349 |          0.086  |          0.2336 |              0.5592 |              0.575  |
| CM       |              0.2403 |                  0.173  |         1.1871 |          1.6002 |        -0.3609 |        -0.017  |         -0.025  |        0.8222 |                 0.1382 |          0.1022 |          0.2498 |              0.5733 |              0.5678 |
| CNQ      |              0.3281 |                  0.304  |         0.9641 |          1.3896 |        -0.2985 |        -0.0316 |         -0.044  |        1.352  |                 0.2047 |          0.1234 |          0.2168 |              0.6594 |              0.5466 |
| ENB      |              0.177  |                  0.1551 |         0.9151 |          1.2191 |        -0.2131 |        -0.0154 |         -0.0234 |        0.675  |                 0.1197 |          0.0573 |          0.2103 |              0.1215 |              0.5622 |
| TD       |              0.1765 |                  0.1678 |         0.8437 |          1.0807 |        -0.2605 |        -0.0165 |         -0.0253 |        0.7659 |                 0.1311 |          0.0454 |          0.1848 |              0.115  |              0.5566 |
| BMO      |              0.1812 |                  0.1778 |         0.8224 |          1.0568 |        -0.2745 |        -0.0176 |         -0.0266 |        0.9517 |                 0.1544 |          0.0268 |          0.1536 |              0.1596 |              0.5463 |
| SU       |              0.2885 |                  0.314  |         0.8074 |          1.197  |        -0.306  |        -0.0304 |         -0.044  |        1.2733 |                 0.1948 |          0.0937 |          0.1991 |              0.4725 |              0.5319 |
| BNS      |              0.144  |                  0.1595 |         0.6831 |          0.9218 |        -0.3489 |        -0.0155 |         -0.0236 |        0.7466 |                 0.1287 |          0.0153 |          0.146  |             -0.1252 |              0.5502 |
| ATD      |              0.1205 |                  0.2355 |         0.3628 |          0.5665 |        -0.2216 |        -0.0222 |         -0.0311 |        0.7118 |                 0.1243 |         -0.0039 |          0.1201 |             -0.1816 |              0.5064 |
| CNR      |              0.0145 |                  0.199  |        -0.103  |         -0.1487 |        -0.2724 |        -0.0192 |         -0.0287 |        0.8345 |                 0.1397 |         -0.1252 |         -0.0246 |             -0.8617 |              0.5136 |
| T        |             -0.0072 |                  0.1589 |        -0.2658 |         -0.3662 |        -0.3567 |        -0.0154 |         -0.0232 |        0.3983 |                 0.085  |         -0.0922 |         -0.106  |             -0.9921 |              0.4976 |
| BCE      |             -0.0323 |                  0.1669 |        -0.4035 |         -0.5067 |        -0.4986 |        -0.015  |         -0.0257 |        0.2642 |                 0.0682 |         -0.1005 |         -0.2549 |             -1.0226 |              0.4896 |

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
