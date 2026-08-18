from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    score = np.zeros_like(denominator, dtype="float64")
    np.divide(diff, denominator, out=score, where=denominator != 0)
    return float(np.mean(score) * 100)


def forecast_bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_pred - y_true).mean())


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true = pd.Series(y_true).astype("float32")
    y_pred = pd.Series(y_pred).astype("float32")

    if len(y_true) != len(y_pred) or y_true.empty:
        raise ValueError("y_true and y_pred must be non-empty and have equal length")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true and y_pred must contain only finite values")

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": forecast_bias(y_true, y_pred),
    }
