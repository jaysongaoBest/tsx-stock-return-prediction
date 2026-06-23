# TSX Financial Risk Analysis and Stock Return Prediction

This project is a Python-based financial analytics portfolio project built from daily TSX equity data. It analyzes stock performance, downside risk, market sensitivity, and short-term return predictability for 12 Canadian-listed companies.

The goal is not to claim a profitable trading strategy. Instead, the project demonstrates how Python can be used in a financial analyst or risk analyst workflow: cleaning market data, calculating risk metrics, visualizing security-level behavior, and testing machine learning models with realistic limitations.

## Questions Answered

This project focuses on four practical analyst questions:

1. Which stocks delivered the strongest risk-adjusted performance?
2. Which stocks carried the highest volatility, drawdown, and tail-risk exposure?
3. How sensitive were individual stocks to the broader TSX market?
4. Can historical returns, volume, volatility, dividends, and market index features help predict next-day return direction?

## Dataset

The dataset contains 15,049 daily observations from 2021-01-04 to 2025-12-31 for 12 TSX tickers:

```text
ATD, BCE, BMO, BNS, CM, CNQ, CNR, ENB, RY, SU, T, TD
```

Key fields include daily closing price, daily volume, daily return, dividend information, and S&P/TSX Composite index levels.

The raw CFMRC/course data file is not included in this public repository because of data access restrictions. The repository includes the full Python code, generated charts, risk metrics, model results, and analyst-style reports.

## Methods

Financial risk analysis:

- Annualized return and annualized volatility
- Sharpe ratio and Sortino ratio
- Maximum drawdown
- Historical Value at Risk (VaR 95%)
- Conditional Value at Risk (CVaR 95%)
- Beta to the S&P/TSX Composite Total Return Index
- CAPM expected return
- Jensen's alpha
- Treynor ratio
- Tracking error and information ratio

Machine learning:

- Feature engineering with lagged returns, rolling returns, rolling volatility, market return, excess return, dividends, and volume
- Linear Regression for next-day return prediction
- Logistic Regression for next-day positive/negative return classification
- Decision Tree for interpretable classification
- Random Forest as a non-linear benchmark
- Time-based train/test split to reduce look-ahead bias

## Key Findings

The risk analysis suggests that **RY** had the strongest risk-adjusted performance in this sample, with the highest Sharpe and Sortino ratios. **CNQ** produced the highest Jensen's alpha and the highest beta, indicating strong outperformance relative to CAPM expectations but also higher market sensitivity. **BCE** had the deepest maximum drawdown and the weakest information ratio, making it one of the weaker risk-adjusted performers in the sample.

Selected results:

| Area | Result |
| --- | --- |
| Best Sharpe Ratio | RY |
| Best Sortino Ratio | RY |
| Highest Jensen's Alpha | CNQ |
| Highest Beta | CNQ |
| Worst Maximum Drawdown | BCE |
| Weakest Information Ratio | BCE |

These results are useful from a financial analyst perspective because they separate absolute return from risk-adjusted return, downside risk, and benchmark-relative performance.

## Machine Learning Results

Using a time-based 80/20 split, the modeling dataset contained 14,767 rows after feature engineering.

| Model | Task | Result |
| --- | --- | --- |
| Linear Regression | Predict next-day return | MAE 0.0092, RMSE 0.0132, R-squared -0.0447 |
| Logistic Regression | Predict positive next-day return | Accuracy 0.4641, ROC AUC 0.5246 |
| Decision Tree | Predict positive next-day return | Accuracy 0.5501 |
| Random Forest | Predict positive next-day return | Accuracy 0.5149, ROC AUC 0.5089 |

The model results show that next-day stock return prediction is difficult and noisy, which is consistent with real-world market behavior. The Decision Tree produced the best classification accuracy in this initial comparison, but the overall results should be interpreted as a machine learning workflow demonstration rather than a deployable trading signal.

## Visual Outputs

The project generates several charts for financial analysis and model interpretation:

- `visuals/indexed_total_return.png`: growth of a $100 investment by ticker
- `visuals/risk_return_scatter.png`: annualized return versus annualized volatility
- `visuals/return_correlation_heatmap.png`: daily return correlation matrix
- `visuals/drawdown_by_ticker.png`: drawdown path by ticker
- `visuals/rolling_volatility_60d.png`: 60-day rolling annualized volatility
- `visuals/var_cvar_by_ticker.png`: historical VaR and CVaR comparison
- `visuals/jensens_alpha_by_ticker.png`: realized return versus CAPM-implied expected return
- `visuals/sortino_ratio_by_ticker.png`: downside-risk-adjusted performance
- `visuals/logistic_confusion_matrix.png`: classification performance for Logistic Regression
- `visuals/decision_tree.png`: interpretable tree model structure
- `visuals/feature_importance.png`: Random Forest feature importance

## Project Structure

```text
tsx-stock-return-prediction/
  README.md
  requirements.txt
  src/
    data_preprocessing.py
    risk_analysis.py
    modeling.py
  reports/
    financial_risk_report.md
    model_results.md
    risk_metrics.csv
  visuals/
    *.png
```

## How to Run

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw dataset in the project root as:

```text
Portfolio Project.csv
```

Run the financial risk analysis:

```bash
python src/risk_analysis.py
```

Run the machine learning models:

```bash
python src/modeling.py
```

## Main Takeaway

This project shows how Python can support an end-to-end financial analysis workflow. It combines market data cleaning, risk measurement, benchmark comparison, financial economics formulas, visualization, and machine learning evaluation. For a financial analyst or risk analyst role, the strongest value of the project is its ability to translate raw equity data into interpretable risk metrics and clear investment insights.
