from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.data.storage import load_parquet, save_parquet
from app.features.calendar_features import add_calendar_features
from app.features.hierarchy_features import add_hierarchy_features
from app.features.lag_features import add_lag_features
from app.features.price_features import add_price_features
from app.features.schema import CATEGORICAL_FEATURE_COLUMNS
from app.utils.paths import get_processed_file_path

logger = get_logger()


IDENTITY_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "date",
    "d",
]

TARGET_COLUMN = "target"


def optimize_base_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    categorical_cols = ["id", *CATEGORICAL_FEATURE_COLUMNS]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    int8_cols = ["snap_CA", "snap_TX", "snap_WI"]
    for col in int8_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("int8")

    if "sales" in df.columns:
        df["sales"] = df["sales"].astype("float32")

    if "sell_price" in df.columns:
        df["sell_price"] = df["sell_price"].astype("float32")

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    return df


def filter_history_window(df: pd.DataFrame) -> pd.DataFrame:
    settings = get_settings()
    df = df.copy()

    if settings.feature_build_start_date:
        start_date = pd.to_datetime(settings.feature_build_start_date)
        df = df[df["date"] >= start_date].copy()
        logger.info("Applied feature_build_start_date filter: {}", start_date.date())
        return df

    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=settings.history_days_for_training - 1)
    df = df[df["date"] >= cutoff_date].copy()
    logger.info(
        "Applied history window filter: keeping rows from {} to {}",
        cutoff_date.date(),
        max_date.date(),
    )
    return df


def add_target_column(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COLUMN] = (
        df.groupby(["store_id", "item_id"], observed=False)["sales"]
        .shift(-horizon_days)
        .astype("float32")
    )
    return df


def finalize_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    training_required_columns = [
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
        "sales_roll_mean_7",
        "sales_roll_mean_28",
        TARGET_COLUMN,
    ]

    df = df.dropna(subset=training_required_columns).reset_index(drop=True)

    feature_cols_to_fill_zero = [
        "sales_roll_std_7",
        "sales_roll_std_28",
        "price_roll_std_7",
    ]
    for col in feature_cols_to_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("float32")

    return df


def build_feature_table_from_base(base_df: pd.DataFrame) -> pd.DataFrame:
    settings = get_settings()

    logger.info("Optimizing base table dtypes")
    df = optimize_base_dtypes(base_df)

    logger.info("Filtering base table to manageable training window")
    df = filter_history_window(df)

    logger.info("Sorting data before time-series feature generation")
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    logger.info("Adding calendar features")
    df = add_calendar_features(df)

    logger.info("Adding price features")
    df = add_price_features(df)

    logger.info("Adding lag and rolling sales features")
    df = add_lag_features(df)

    logger.info("Adding hierarchy aggregate features")
    df = add_hierarchy_features(df)

    logger.info("Adding target column for horizon={}", settings.forecast_horizon_days)
    df = add_target_column(df, horizon_days=settings.forecast_horizon_days)

    logger.info("Finalizing feature table")
    df = finalize_feature_table(df)

    logger.info("Feature table built successfully | shape={}", df.shape)
    return df


def build_and_save_feature_table() -> Path:
    settings = get_settings()

    base_path = get_processed_file_path(settings.processed_base_table_name)
    feature_path = get_processed_file_path(settings.feature_table_name)

    logger.info("Loading base table from {}", base_path)
    base_df = load_parquet(base_path)

    feature_df = build_feature_table_from_base(base_df)

    logger.info("Saving feature table to {}", feature_path)
    save_parquet(feature_df, feature_path)

    return feature_path
