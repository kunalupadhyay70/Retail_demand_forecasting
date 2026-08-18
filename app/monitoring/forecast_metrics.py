from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.data.storage import load_parquet
from app.training.evaluate import evaluate_predictions
from app.utils.common import utc_timestamp
from app.utils.database import insert_monitoring_run
from app.utils.paths import get_processed_file_path

logger = get_logger()


def load_latest_prediction_file() -> Path:
    settings = get_settings()
    prediction_files = sorted(settings.predictions_dir.glob("batch_forecast_*.parquet"))

    if not prediction_files:
        raise FileNotFoundError(
            "No batch forecast parquet files found in predictions directory"
        )

    return prediction_files[-1]


def load_actuals_base_table() -> pd.DataFrame:
    settings = get_settings()
    base_path = get_processed_file_path(settings.processed_base_table_name)
    logger.info("Loading base table for actuals from {}", base_path)

    actual_df = load_parquet(base_path)

    actual_df = actual_df[["item_id", "store_id", "date", "sales"]].copy()
    actual_df["date"] = pd.to_datetime(actual_df["date"])
    actual_df["sales"] = actual_df["sales"].astype("float32")

    return actual_df


def evaluate_prediction_file_against_actuals(
    prediction_file: Path,
    actual_df: pd.DataFrame,
) -> dict:
    pred_df = load_parquet(prediction_file).copy()
    pred_df["target_date"] = pd.to_datetime(pred_df["target_date"])
    pred_df["predicted_sales"] = pred_df["predicted_sales"].astype("float32")

    merged = pred_df.merge(
        actual_df,
        left_on=["item_id", "store_id", "target_date"],
        right_on=["item_id", "store_id", "date"],
        how="left",
    )

    merged = merged.dropna(subset=["sales", "predicted_sales"]).reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            "No matching actuals found for prediction file evaluation. "
            "Check whether target_date exists in the base table."
        )

    metrics = evaluate_predictions(
        y_true=merged["sales"],
        y_pred=merged["predicted_sales"],
    )

    result = {
        "run_timestamp": utc_timestamp(),
        "prediction_file": str(prediction_file),
        "joined_row_count": len(merged),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "smape": metrics["smape"],
        "bias": metrics["bias"],
    }

    insert_monitoring_run(result)
    logger.info("Forecast monitoring evaluation completed | {}", result)
    return result
