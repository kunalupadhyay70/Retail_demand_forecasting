from __future__ import annotations

import pandas as pd

from app.config.logging_config import get_logger
from app.utils.common import utc_timestamp
from app.utils.database import insert_data_quality_run

logger = get_logger()


def run_data_quality_checks(df: pd.DataFrame, dataset_name: str) -> dict:
    row_count = len(df)
    missing_value_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    result = {
        "run_timestamp": utc_timestamp(),
        "dataset_name": dataset_name,
        "row_count": row_count,
        "missing_value_count": missing_value_count,
        "duplicate_count": duplicate_count,
    }

    insert_data_quality_run(result)
    logger.info("Data quality check completed for {} | {}", dataset_name, result)
    return result
