# Model Results

The script `src/modeling.py` was run successfully on the local dataset.

## Dataset

- Rows after feature engineering: 14,767
- Training rows: 11,815
- Testing rows: 2,952
- Split method: time-based 80/20 split

## Linear Regression

Task: predict next-day return.

- MAE: 0.009210
- RMSE: 0.013233
- R-squared: -0.0447

## Logistic Regression

Task: classify whether next-day return is positive.

- Accuracy: 0.4641
- ROC AUC: 0.5246

## Decision Tree

Task: classify whether next-day return is positive.

- Accuracy: 0.5501

## Random Forest

Task: classify whether next-day return is positive.

- Accuracy: 0.5149
- ROC AUC: 0.5089

## Interpretation

The models demonstrate a complete machine learning workflow, but the predictive signal is limited. This is expected for short-term public market data, where next-day returns are noisy and difficult to forecast. The strongest portfolio value of this project is the end-to-end workflow: cleaning raw financial data, building time-aware features, training several model families, evaluating results, and interpreting model limitations.
