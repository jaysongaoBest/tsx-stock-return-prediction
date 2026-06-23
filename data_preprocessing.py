"""Data loading and feature engineering for the TSX portfolio project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = {
    "trdate-Trade Date": "trade_date",
    "usage-Usage Number": "usage_number",
    "symbol-Ticker": "ticker",
    "name-Name": "company_name",
    "business-Business": "business",
    "stock_type-Stock Type": "stock_type",
    "closeprice-Daily Closing price": "close_price",
    "volume-Daily Volume": "volume",
    "return-Daily Return": "daily_return",
    "dividend-Dividend": "dividend",
    "trdate-Ex-dividend date": "ex_dividend_date",
    "ind1-S&P/TSX Composite Daily Price Index": "tsx_price_index",
    "ind2-S&P/TSX Composite Daily Total Return Index": "tsx_total_return_index",
}


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the CFMRC CSV export and normalize column names."""
    df = pd.read_csv(path, skiprows=1)
    df = df.rename(columns=RAW_COLUMNS)

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["ex_dividend_date"] = pd.to_datetime(df["ex_dividend_date"], errors="coerce")

    numeric_columns = [
        "close_price",
        "volume",
        "daily_return",
        "dividend",
        "tsx_price_index",
        "tsx_total_return_index",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged, rolling, and market-relative features for modeling."""
    featured = df.copy()
    grouped = featured.groupby("ticker", group_keys=False)

    featured["return_lag_1"] = grouped["daily_return"].shift(1)
    featured["return_lag_5"] = grouped["daily_return"].shift(5)
    featured["volume_lag_1"] = grouped["volume"].shift(1)
    featured["price_lag_1"] = grouped["close_price"].shift(1)

    featured["rolling_return_5"] = grouped["daily_return"].rolling(5).mean().reset_index(level=0, drop=True)
    featured["rolling_return_20"] = grouped["daily_return"].rolling(20).mean().reset_index(level=0, drop=True)
    featured["rolling_volatility_20"] = grouped["daily_return"].rolling(20).std().reset_index(level=0, drop=True)
    featured["rolling_volume_20"] = grouped["volume"].rolling(20).mean().reset_index(level=0, drop=True)

    featured["market_return"] = featured.groupby("ticker")["tsx_total_return_index"].pct_change()
    featured["excess_return"] = featured["daily_return"] - featured["market_return"]
    featured["dividend_paid"] = featured["dividend"].fillna(0).gt(0).astype(int)
    featured["log_volume"] = np.log1p(featured["volume"])

    featured["target_next_return"] = grouped["daily_return"].shift(-1)
    featured["target_next_positive"] = featured["target_next_return"].gt(0).astype(int)

    return featured


def build_modeling_dataset(path: str | Path) -> pd.DataFrame:
    """Return a clean modeling table ready for scikit-learn."""
    df = add_features(load_raw_data(path))
    feature_columns = get_feature_columns()
    required = feature_columns + ["target_next_return", "target_next_positive"]
    return df.dropna(subset=required).reset_index(drop=True)


def get_feature_columns() -> list[str]:
    """Central list of model input features."""
    return [
        "close_price",
        "log_volume",
        "daily_return",
        "return_lag_1",
        "return_lag_5",
        "rolling_return_5",
        "rolling_return_20",
        "rolling_volatility_20",
        "rolling_volume_20",
        "market_return",
        "excess_return",
        "dividend_paid",
        "tsx_price_index",
        "tsx_total_return_index",
    ]
