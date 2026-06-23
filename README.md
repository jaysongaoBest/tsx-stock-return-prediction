# TSX Financial Risk and Stock Return Prediction Portfolio Project

This project upgrades a course-style R/data assignment into a Python financial analysis and machine learning portfolio project. It uses daily Canadian equity data from the CFMRC Securities Daily Tables to analyze stock performance, measure downside risk, visualize market behavior, and predict next-day return direction.

## Business Question

How can Python be used to compare TSX-listed stocks from a financial analyst/risk analyst perspective, and can recent price movement, volume, dividends, and market index behavior help predict next-day return direction?

## Dataset

The dataset contains 15,049 daily observations from 2021-01-04 to 2025-12-31 for 12 TSX tickers, including ATD, BCE, BMO, BNS, CM, CNQ, CNR, ENB, RY, SU, T, and TD.

Key fields include:

- Daily closing price
- Daily trading volume
- Daily stock return
- Dividend amount and ex-dividend date
- S&P/TSX Composite Daily Price Index
- S&P/TSX Composite Daily Total Return Index

## Methods Used

- Python data cleaning with pandas
- Exploratory data analysis and visualization with matplotlib/seaborn
- Financial economics and risk analysis using annualized return, annualized volatility, Sharpe ratio, Sortino ratio, beta, CAPM expected return, Jensen's alpha, Treynor ratio, tracking error, information ratio, maximum drawdown, historical VaR, and historical CVaR
- Feature engineering with lagged returns, rolling averages, volatility, market return, and excess return
- Linear Regression for next-day return prediction
- Logistic Regression for next-day positive/negative return classification
- Decision Tree Classifier for interpretable classification
- Random Forest Classifier for non-linear benchmark performance
- Time-based train/test split to reduce look-ahead bias

## Project Structure

```text
portfolio project/
  README.md
  requirements.txt
  src/
    data_preprocessing.py
    risk_analysis.py
    modeling.py
  visuals/
  models/
  reports/
  notebooks/
```

Note: the raw CFMRC/course data file is excluded from version control by default. To run the project locally, place `Portfolio Project.csv` in the project root.

## How to Run

Create a virtual environment and install the dependencies:

```bash
cd "/Users/jaysongao/Desktop/portfolio project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the models and generate visuals:

```bash
python src/risk_analysis.py
python src/modeling.py
```

Outputs are saved in:

- `visuals/` for charts and model interpretation images
- `reports/` for risk metrics and written analysis
- `models/` for trained `.joblib` model files

## Financial Risk Analysis Outputs

- `visuals/indexed_total_return.png`: growth of a $100 investment by ticker
- `visuals/risk_return_scatter.png`: annualized return versus annualized volatility
- `visuals/return_correlation_heatmap.png`: return correlation matrix
- `visuals/drawdown_by_ticker.png`: drawdown by ticker
- `visuals/rolling_volatility_60d.png`: 60-day rolling annualized volatility
- `visuals/var_cvar_by_ticker.png`: historical VaR and CVaR comparison
- `visuals/jensens_alpha_by_ticker.png`: realized return versus CAPM-implied expected return
- `visuals/sortino_ratio_by_ticker.png`: downside-risk-adjusted performance
- `reports/risk_metrics.csv`: risk metrics table
- `reports/financial_risk_report.md`: written analyst-style report

## Financial Economics Formulas

This project uses formulas commonly seen in financial analyst, portfolio risk, and investment analysis workflows:

```text
CAPM Expected Return = Risk-Free Rate + Beta * (Market Return - Risk-Free Rate)
Jensen's Alpha = Realized Annualized Return - CAPM Expected Return
Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Annualized Volatility
Sortino Ratio = (Annualized Return - Risk-Free Rate) / Downside Deviation
Treynor Ratio = (Annualized Return - Risk-Free Rate) / Beta
Historical VaR 95% = 5th Percentile of Daily Returns
Historical CVaR 95% = Average Return Conditional on Return <= VaR 95%
Information Ratio = Active Return / Tracking Error
```

WACC is discussed in the analyst report as a corporate finance extension. It is not calculated directly because this dataset contains market prices and returns, not debt, equity value, tax rate, or cost of debt inputs.

## Portfolio Summary

Built a Python-based financial risk analytics and machine learning project using daily TSX stock data. Cleaned and engineered time-series features, calculated annualized return, volatility, Sharpe ratio, Sortino ratio, beta, CAPM expected return, Jensen's alpha, Treynor ratio, tracking error, information ratio, maximum drawdown, VaR, and CVaR, then trained Linear Regression, Logistic Regression, Decision Tree, and Random Forest models to predict next-day stock return behavior. Evaluated model performance with regression and classification metrics, including MAE, RMSE, R-squared, accuracy, ROC AUC, and confusion matrices.

## Initial Model Results

Using a time-based 80/20 split, the modeling dataset contained 14,767 rows after feature engineering.

| Model | Task | Key Result |
| --- | --- | --- |
| Linear Regression | Predict next-day return | MAE 0.0092, RMSE 0.0132, R-squared -0.0447 |
| Logistic Regression | Predict positive next-day return | Accuracy 0.4641, ROC AUC 0.5246 |
| Decision Tree | Predict positive next-day return | Accuracy 0.5501 |
| Random Forest | Predict positive next-day return | Accuracy 0.5149, ROC AUC 0.5089 |

The results show that short-term stock return prediction is noisy and difficult, which is consistent with financial market behavior. The project is therefore framed as a machine learning workflow and model comparison study, rather than a trading strategy.

## Resume Bullet

Developed a Python financial risk analytics and machine learning pipeline for TSX equities using pandas, seaborn, and scikit-learn; calculated CAPM expected return, Jensen's alpha, Sharpe ratio, Sortino ratio, Treynor ratio, beta, maximum drawdown, VaR, and CVaR, engineered lagged return and rolling volatility features, and compared Linear Regression, Logistic Regression, Decision Tree, and Random Forest models using time-based validation.
