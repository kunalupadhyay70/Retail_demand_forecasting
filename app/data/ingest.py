from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.utils.paths import get_raw_file_path

logger = get_logger()


def _read_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")

    logger.info(f"Reading file: {file_path}")
    return pd.read_csv(file_path)


def load_sales_data() -> pd.DataFrame:
    settings = get_settings()
    return _read_csv(get_raw_file_path(settings.sales_file_name))


def load_calendar_data() -> pd.DataFrame:
    settings = get_settings()
    return _read_csv(get_raw_file_path(settings.calendar_file_name))


def load_prices_data() -> pd.DataFrame:
    settings = get_settings()
    return _read_csv(get_raw_file_path(settings.prices_file_name))


def load_raw_m5_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales_df = load_sales_data()
    calendar_df = load_calendar_data()
    prices_df = load_prices_data()

    logger.info(
        "Loaded raw datasets | sales shape={} | calendar shape={} | prices shape={}",
        sales_df.shape,
        calendar_df.shape,
        prices_df.shape,
    )

    return sales_df, calendar_df, prices_df
