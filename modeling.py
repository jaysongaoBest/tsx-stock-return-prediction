"""Train baseline machine learning models for the TSX portfolio project."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import build_modeling_dataset, get_feature_columns


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "Portfolio Project.csv"
VISUALS_DIR = PROJECT_DIR / "visuals"
MODELS_DIR = PROJECT_DIR / "models"


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use earlier dates for training and later dates for testing."""
    cutoff = df["trade_date"].quantile(0.8)
    train = df[df["trade_date"] <= cutoff].copy()
    test = df[df["trade_date"] > cutoff].copy()
    return train, test


def make_numeric_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", ColumnTransformer([("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), get_feature_columns())])),
            ("model", model),
        ]
    )


def train_models() -> None:
    VISUALS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df = build_modeling_dataset(DATA_PATH)
    train, test = time_based_split(df)
    features = get_feature_columns()

    x_train = train[features]
    x_test = test[features]
    y_train_reg = train["target_next_return"]
    y_test_reg = test["target_next_return"]
    y_train_cls = train["target_next_positive"]
    y_test_cls = test["target_next_positive"]

    linear_model = make_numeric_pipeline(LinearRegression())
    logistic_model = make_numeric_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    tree_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
    forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=40,
        random_state=42,
        class_weight="balanced",
    )

    linear_model.fit(x_train, y_train_reg)
    logistic_model.fit(x_train, y_train_cls)
    tree_model.fit(x_train, y_train_cls)
    forest_model.fit(x_train, y_train_cls)

    reg_predictions = linear_model.predict(x_test)
    log_predictions = logistic_model.predict(x_test)
    log_probabilities = logistic_model.predict_proba(x_test)[:, 1]
    tree_predictions = tree_model.predict(x_test)
    forest_predictions = forest_model.predict(x_test)
    forest_probabilities = forest_model.predict_proba(x_test)[:, 1]

    print("Dataset")
    print(f"Rows after feature engineering: {len(df):,}")
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    print()

    print("Linear Regression: next-day return prediction")
    print(f"MAE:  {mean_absolute_error(y_test_reg, reg_predictions):.6f}")
    print(f"RMSE: {root_mean_squared_error(y_test_reg, reg_predictions):.6f}")
    print(f"R2:   {r2_score(y_test_reg, reg_predictions):.4f}")
    print()

    print("Logistic Regression: next-day positive return classification")
    print(f"Accuracy: {accuracy_score(y_test_cls, log_predictions):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test_cls, log_probabilities):.4f}")
    print(classification_report(y_test_cls, log_predictions))

    print("Decision Tree: next-day positive return classification")
    print(f"Accuracy: {accuracy_score(y_test_cls, tree_predictions):.4f}")
    print(classification_report(y_test_cls, tree_predictions))

    print("Random Forest: next-day positive return classification")
    print(f"Accuracy: {accuracy_score(y_test_cls, forest_predictions):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test_cls, forest_probabilities):.4f}")
    print(classification_report(y_test_cls, forest_predictions))

    plot_outputs(df, y_test_cls, log_predictions, tree_model, forest_model, features)

    joblib.dump(linear_model, MODELS_DIR / "linear_regression.joblib")
    joblib.dump(logistic_model, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(tree_model, MODELS_DIR / "decision_tree.joblib")
    joblib.dump(forest_model, MODELS_DIR / "random_forest.joblib")


def plot_outputs(
    df: pd.DataFrame,
    y_test_cls: pd.Series,
    log_predictions,
    tree_model: DecisionTreeClassifier,
    forest_model: RandomForestClassifier,
    features: list[str],
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="trade_date", y="close_price", hue="ticker", legend=False, alpha=0.7)
    plt.title("TSX Stock Closing Prices")
    plt.xlabel("Trade date")
    plt.ylabel("Closing price")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "closing_prices.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test_cls, log_predictions, ax=ax, colorbar=False)
    ax.set_title("Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "logistic_confusion_matrix.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(18, 8))
    plot_tree(tree_model, feature_names=features, class_names=["Down/Flat", "Up"], filled=True, max_depth=3)
    plt.title("Decision Tree for Next-Day Return Direction")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "decision_tree.png", dpi=160)
    plt.close()

    importance = pd.DataFrame({
        "feature": features,
        "importance": forest_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance.head(10), x="importance", y="feature", color="#4C78A8")
    plt.title("Top Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "feature_importance.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    train_models()
