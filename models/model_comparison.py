from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor

    BOOSTING_REGRESSOR = XGBRegressor(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    BOOSTING_CLASSIFIER = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    BOOSTING_NAME = "XGBoost"
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    BOOSTING_REGRESSOR = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
    )
    BOOSTING_CLASSIFIER = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
    )
    BOOSTING_NAME = "Gradient Boosting"


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "modeling_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models" / "saved"


REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42, min_samples_leaf=2
    ),
    BOOSTING_NAME: BOOSTING_REGRESSOR,
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, min_samples_leaf=2
    ),
    BOOSTING_NAME: BOOSTING_CLASSIFIER,
}


@dataclass
class ModelResult:
    task: str
    target: str
    model_name: str
    metrics: dict[str, float]
    recommended: bool = False


def build_preprocessor(feature_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] == 2024].copy()
    test = df[df["year"] == 2025].copy()
    return train, test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100)
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4), "mape": round(mape, 2)}


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None
) -> dict[str, float]:
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


def train_regression_task(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    categorical_cols: list[str],
    task_name: str,
) -> tuple[list[ModelResult], Pipeline | None, pd.DataFrame]:
    train, test = time_split(df)
    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    results: list[ModelResult] = []
    best_model: Pipeline | None = None
    best_r2 = -np.inf
    forecast_rows: list[dict[str, Any]] = []

    for model_name, estimator in REGRESSION_MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_cols, categorical_cols)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = regression_metrics(y_test.to_numpy(), predictions)
        results.append(
            ModelResult(
                task=task_name,
                target=target,
                model_name=model_name,
                metrics=metrics,
            )
        )

        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_model = pipeline

        if model_name == BOOSTING_NAME:
            for actual, predicted, record_date in zip(
                y_test, predictions, test["date"]
            ):
                forecast_rows.append(
                    {
                        "date": pd.Timestamp(record_date).strftime("%Y-%m-%d"),
                        "target": target,
                        "actual": float(actual),
                        "predicted": float(predicted),
                    }
                )

    best = max(results, key=lambda item: item.metrics["r2"])
    for result in results:
        result.recommended = result.model_name == best.model_name

    forecast_df = pd.DataFrame(forecast_rows)
    return results, best_model, forecast_df


def train_classification_task(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    categorical_cols: list[str],
    task_name: str,
) -> tuple[list[ModelResult], Pipeline | None]:
    train, test = time_split(df)
    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    results: list[ModelResult] = []
    best_model: Pipeline | None = None
    best_f1 = -np.inf

    for model_name, estimator in CLASSIFICATION_MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_cols, categorical_cols)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = (
            pipeline.predict_proba(X_test)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else None
        )
        metrics = classification_metrics(y_test.to_numpy(), predictions, probabilities)
        results.append(
            ModelResult(
                task=task_name,
                target=target,
                model_name=model_name,
                metrics=metrics,
            )
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = pipeline

    best = max(results, key=lambda item: item.metrics["f1"])
    for result in results:
        result.recommended = result.model_name == best.model_name

    return results, best_model


def extract_feature_importance(
    pipeline: Pipeline, feature_cols: list[str], categorical_cols: list[str]
) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = list(preprocessor.get_feature_names_out())
    if hasattr(model, "feature_importances_"):
        values = np.ravel(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    size = min(len(feature_names), len(values))
    importance = pd.DataFrame(
        {"feature": feature_names[:size], "importance": values[:size]}
    )
    importance = importance.sort_values("importance", ascending=False).head(15)
    importance["importance"] = importance["importance"].round(4)
    return importance.reset_index(drop=True)


def train_all_models() -> dict[str, Any]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    donation_features = [
        "month_num",
        "is_festive_season",
        "city",
        "program",
        "donor_type",
        "channel",
        "volunteer_hours",
        "event_count",
        "social_reach",
        "engagement_rate",
        "donation_lag_1",
        "donation_lag_2",
        "donation_lag_3",
        "literacy_rate_pct",
        "multidimensional_poverty_index",
    ]
    beneficiary_features = [
        "month_num",
        "is_festive_season",
        "city",
        "program",
        "expense_amount",
        "volunteer_hours",
        "event_count",
        "social_reach",
        "engagement_rate",
        "beneficiaries_lag_1",
        "beneficiaries_lag_2",
        "literacy_rate_pct",
        "multidimensional_poverty_index",
    ]
    efficiency_features = [
        "month_num",
        "is_festive_season",
        "city",
        "program",
        "donor_type",
        "channel",
        "expense_amount",
        "volunteer_hours",
        "event_count",
        "social_reach",
        "literacy_rate_pct",
        "multidimensional_poverty_index",
    ]

    donation_categorical = ["city", "program", "donor_type", "channel"]
    beneficiary_categorical = ["city", "program"]
    efficiency_categorical = ["city", "program", "donor_type", "channel"]

    donation_results, donation_model, forecast_df = train_regression_task(
        df,
        target="donation_amount",
        feature_cols=donation_features,
        categorical_cols=donation_categorical,
        task_name="Donation Forecasting",
    )
    beneficiary_results, beneficiary_model, _ = train_regression_task(
        df,
        target="beneficiaries_reached",
        feature_cols=beneficiary_features,
        categorical_cols=beneficiary_categorical,
        task_name="Beneficiary Reach Prediction",
    )
    efficiency_results, efficiency_model = train_classification_task(
        df,
        target="high_efficiency",
        feature_cols=efficiency_features,
        categorical_cols=efficiency_categorical,
        task_name="Campaign Efficiency Classification",
    )

    donation_importance = extract_feature_importance(
        donation_model, donation_features, donation_categorical
    )
    beneficiary_importance = extract_feature_importance(
        beneficiary_model, beneficiary_features, beneficiary_categorical
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    forecast_df.to_csv(OUTPUT_DIR / "forecast_vs_actual.csv", index=False)
    donation_importance.to_csv(OUTPUT_DIR / "donation_feature_importance.csv", index=False)
    beneficiary_importance.to_csv(
        OUTPUT_DIR / "beneficiary_feature_importance.csv", index=False
    )

    joblib.dump(donation_model, MODEL_DIR / "donation_model.joblib")
    joblib.dump(beneficiary_model, MODEL_DIR / "beneficiary_model.joblib")
    joblib.dump(efficiency_model, MODEL_DIR / "efficiency_model.joblib")

    all_results = donation_results + beneficiary_results + efficiency_results
    payload = {
        "results": [asdict(result) for result in all_results],
        "recommended_models": {
            "donation_forecasting": next(
                item.model_name for item in donation_results if item.recommended
            ),
            "beneficiary_reach": next(
                item.model_name for item in beneficiary_results if item.recommended
            ),
            "campaign_efficiency": next(
                item.model_name for item in efficiency_results if item.recommended
            ),
        },
    }

    with (OUTPUT_DIR / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload
