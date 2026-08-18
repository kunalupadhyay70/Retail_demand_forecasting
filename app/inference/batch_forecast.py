from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.data.storage import save_parquet
from app.inference.prepare_features import (
    align_categorical_features,
    get_model_categorical_schema,
)
from app.utils.common import utc_filename_timestamp, utc_timestamp
from app.utils.database import insert_batch_forecast_run

logger = get_logger()


def build_latest_item_store_index(feature_df: pd.DataFrame) -> pd.DataFrame:
    latest_rows = (
        feature_df.sort_values(["store_id", "item_id", "date"])
        .groupby(["store_id", "item_id"], as_index=False, observed=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest_rows


def generate_batch_predictions_from_latest_rows(
    latest_rows_df: pd.DataFrame,
    model_bundle: dict,
) -> pd.DataFrame:
    settings = get_settings()

    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    feature_columns = metadata["feature_columns"]

    missing_cols = [col for col in feature_columns if col not in latest_rows_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns for batch inference: {missing_cols}"
        )

    X = align_categorical_features(
        latest_rows_df[feature_columns].copy(),
        get_model_categorical_schema(model_bundle),
    )
    preds = np.clip(model.predict(X), a_min=0, a_max=None)

    result_df = latest_rows_df[["item_id", "store_id", "date"]].copy()

    result_df["feature_row_date"] = pd.to_datetime(result_df["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    result_df["target_date"] = (
        pd.to_datetime(result_df["date"])
        + pd.Timedelta(days=settings.forecast_horizon_days)
    ).dt.strftime("%Y-%m-%d")
    result_df["forecast_horizon_days"] = settings.forecast_horizon_days
    result_df["predicted_sales"] = preds.astype("float32")
    result_df["model_name"] = metadata.get("model_name", "lightgbm_model")

    metrics = metadata.get("metrics", {})
    result_df["rmse"] = metrics.get("rmse")
    result_df["mae"] = metrics.get("mae")
    result_df["smape"] = metrics.get("smape")
    result_df["bias"] = metrics.get("bias")

    result_df = result_df.drop(columns=["date"]).reset_index(drop=True)
    return result_df


def run_batch_forecast(
    model_bundle: dict,
    feature_df: pd.DataFrame,
    limit: int | None = None,
) -> Path:
    settings = get_settings()

    if limit is None:
        limit = settings.max_batch_forecast_rows
    if limit < 1 or limit > settings.max_batch_forecast_rows:
        raise ValueError(
            f"limit must be between 1 and {settings.max_batch_forecast_rows}"
        )

    latest_index_df = build_latest_item_store_index(feature_df)

    latest_index_df = latest_index_df.head(limit).copy()

    logger.info("Running batch forecast for {} item-store pairs", len(latest_index_df))

    result_df = generate_batch_predictions_from_latest_rows(
        latest_rows_df=latest_index_df,
        model_bundle=model_bundle,
    )

    run_ts = utc_filename_timestamp()
    output_path = settings.predictions_dir / f"batch_forecast_{run_ts}.parquet"

    save_parquet(result_df, output_path)

    insert_batch_forecast_run(
        {
            "run_timestamp": utc_timestamp(),
            "output_path": str(output_path),
            "row_count": len(result_df),
            "model_name": model_bundle["metadata"].get("model_name", "lightgbm_model"),
        }
    )

    logger.info(
        "Batch forecast completed | rows={} | output={}", len(result_df), output_path
    )
    return output_path
