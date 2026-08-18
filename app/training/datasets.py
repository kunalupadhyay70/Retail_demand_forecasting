from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.data.storage import load_parquet
from app.utils.paths import get_processed_file_path

logger = get_logger()


@dataclass
class DatasetSplit:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    feature_columns: list[str]
    target_column: str


EXCLUDE_COLUMNS = {
    "id",
    "d",
    "date",
    "sales",
    "target",
}


def load_feature_table() -> pd.DataFrame:
    settings = get_settings()
    feature_path = get_processed_file_path(settings.feature_table_name)
    logger.info("Loading feature table from {}", feature_path)
    df = load_parquet(feature_path)

    if not pd.api.types.is_datetime64_any_dtype(df[settings.feature_date_column]):
        df[settings.feature_date_column] = pd.to_datetime(
            df[settings.feature_date_column]
        )

    logger.info("Loaded feature table | shape={}", df.shape)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    return feature_columns


def build_train_valid_split(df: pd.DataFrame) -> DatasetSplit:
    settings = get_settings()

    max_date = df[settings.feature_date_column].max()
    valid_start_date = max_date - pd.Timedelta(days=settings.train_validation_days - 1)

    train_df = df[df[settings.feature_date_column] < valid_start_date].copy()
    valid_df = df[df[settings.feature_date_column] >= valid_start_date].copy()

    feature_columns = get_feature_columns(df)

    logger.info(
        "Built train/validation split | train shape={} | valid shape={} | valid_start_date={}",
        train_df.shape,
        valid_df.shape,
        valid_start_date.date(),
    )

    if train_df.empty or valid_df.empty:
        raise ValueError(
            "Train or validation split is empty. Check date filtering logic."
        )

    return DatasetSplit(
        train_df=train_df,
        valid_df=valid_df,
        feature_columns=feature_columns,
        target_column=settings.target_column,
    )
