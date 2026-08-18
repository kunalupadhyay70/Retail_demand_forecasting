from __future__ import annotations

import pandas as pd

from app.config.logging_config import get_logger

logger = get_logger()


def predict_naive_lag_28(valid_df: pd.DataFrame) -> pd.Series:
    if "sales_lag_28" not in valid_df.columns:
        raise ValueError("sales_lag_28 is required for the lag-28 baseline")
    return valid_df["sales_lag_28"].astype("float32")


def predict_rolling_mean_7(valid_df: pd.DataFrame) -> pd.Series:
    if "sales_roll_mean_7" not in valid_df.columns:
        raise ValueError("sales_roll_mean_7 is required for the rolling-mean baseline")
    return valid_df["sales_roll_mean_7"].astype("float32")


def predict_rolling_mean_28(valid_df: pd.DataFrame) -> pd.Series:
    if "sales_roll_mean_28" not in valid_df.columns:
        raise ValueError("sales_roll_mean_28 is required for the rolling-mean baseline")
    return valid_df["sales_roll_mean_28"].astype("float32")


def run_baseline_models(valid_df: pd.DataFrame) -> dict[str, pd.Series]:
    logger.info("Running baseline models")

    predictions = {
        "baseline_lag_28": predict_naive_lag_28(valid_df),
        "baseline_roll_mean_7": predict_rolling_mean_7(valid_df),
        "baseline_roll_mean_28": predict_rolling_mean_28(valid_df),
    }

    logger.info("Baseline predictions generated: {}", list(predictions.keys()))
    return predictions
