from __future__ import annotations

from datetime import date

import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.data.storage import load_parquet
from app.features.schema import CATEGORICAL_FEATURE_COLUMNS
from app.utils.paths import get_processed_file_path

logger = get_logger()


def load_feature_table_for_inference() -> pd.DataFrame:
    settings = get_settings()
    feature_path = get_processed_file_path(settings.feature_table_name)

    logger.info("Loading feature table for inference from {}", feature_path)
    df = load_parquet(feature_path)

    if not pd.api.types.is_datetime64_any_dtype(df[settings.feature_date_column]):
        df[settings.feature_date_column] = pd.to_datetime(
            df[settings.feature_date_column]
        )

    return df


def get_latest_feature_row(
    feature_df: pd.DataFrame,
    item_id: str,
    store_id: str,
    forecast_date: str | date | None = None,
) -> pd.DataFrame:
    settings = get_settings()

    filtered = feature_df[
        (feature_df["item_id"] == item_id) & (feature_df["store_id"] == store_id)
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No feature rows found for item_id={item_id} and store_id={store_id}"
        )

    if forecast_date:
        forecast_date_ts = pd.to_datetime(forecast_date)
        required_feature_date = forecast_date_ts - pd.Timedelta(
            days=settings.forecast_horizon_days
        )
        filtered = filtered[
            filtered[settings.feature_date_column] == required_feature_date
        ].copy()

        if filtered.empty:
            raise ValueError(
                f"No feature row can produce forecast_date={forecast_date} "
                f"for item_id={item_id} and store_id={store_id}"
            )

    latest_row = (
        filtered.sort_values(settings.feature_date_column)
        .tail(1)
        .reset_index(drop=True)
    )

    logger.info(
        "Prepared latest feature row for inference | item_id={} | store_id={} | date={}",
        item_id,
        store_id,
        latest_row.loc[0, settings.feature_date_column],
    )

    return latest_row


def get_model_categorical_schema(model_bundle: dict) -> dict[str, list[str]]:
    """Load the persisted schema, with backward compatibility for old bundles."""
    metadata = model_bundle["metadata"]
    persisted_schema = metadata.get("categorical_schema")
    if persisted_schema is not None:
        return persisted_schema

    feature_columns = metadata.get("feature_columns", [])
    categorical_columns = [
        col for col in CATEGORICAL_FEATURE_COLUMNS if col in feature_columns
    ]
    trained_categories = model_bundle["model"].booster_.pandas_categorical or []
    if len(categorical_columns) != len(trained_categories):
        raise ValueError("Model bundle does not contain a valid categorical schema")
    return {
        column: [str(value) for value in categories]
        for column, categories in zip(
            categorical_columns, trained_categories, strict=True
        )
    }


def align_categorical_features(
    model_input: pd.DataFrame, categorical_schema: dict[str, list[str]]
) -> pd.DataFrame:
    """Restore the categorical schema LightGBM recorded during training.

    Parquet reads all-null categorical columns back as ``object``. LightGBM
    still expects those columns in its pandas categorical metadata, so every
    inference frame must be aligned explicitly before prediction.
    """
    expected_columns = [
        col for col in CATEGORICAL_FEATURE_COLUMNS if col in model_input.columns
    ]
    if expected_columns != list(categorical_schema):
        raise ValueError("Model categorical schema does not match inference features")

    model_input = model_input.copy()
    for column, categories in categorical_schema.items():
        model_input[column] = pd.Categorical(model_input[column], categories=categories)

    return model_input


def extract_model_input(
    feature_row: pd.DataFrame,
    feature_columns: list[str],
    categorical_schema: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    missing_columns = [col for col in feature_columns if col not in feature_row.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required feature columns for inference: {missing_columns}"
        )

    model_input = feature_row[feature_columns].copy()
    if categorical_schema is not None:
        model_input = align_categorical_features(model_input, categorical_schema)
    return model_input
