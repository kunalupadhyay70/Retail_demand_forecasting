from __future__ import annotations

import pandas as pd

from app.config.logging_config import get_logger

logger = get_logger()


SALES_REQUIRED_COLUMNS = {
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
}

CALENDAR_REQUIRED_COLUMNS = {
    "date",
    "wm_yr_wk",
    "d",
    "weekday",
    "month",
    "year",
}

PRICES_REQUIRED_COLUMNS = {
    "store_id",
    "item_id",
    "wm_yr_wk",
    "sell_price",
}


def _check_required_columns(
    df: pd.DataFrame, required_columns: set[str], df_name: str
) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def validate_sales_data(sales_df: pd.DataFrame) -> None:
    _check_required_columns(sales_df, SALES_REQUIRED_COLUMNS, "sales_df")

    day_columns = [col for col in sales_df.columns if col.startswith("d_")]
    if not day_columns:
        raise ValueError("sales_df does not contain day columns like d_1, d_2, ...")

    if sales_df[list(SALES_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("sales_df contains missing identifier values")

    if sales_df[day_columns].isna().any().any():
        raise ValueError("sales_df contains missing sales values")

    if sales_df.duplicated(subset=["id"]).any():
        raise ValueError("sales_df contains duplicate ids")

    if (sales_df[day_columns] < 0).any().any():
        raise ValueError("sales_df contains negative sales values")

    logger.info("sales_df validation passed")


def validate_calendar_data(calendar_df: pd.DataFrame) -> None:
    _check_required_columns(calendar_df, CALENDAR_REQUIRED_COLUMNS, "calendar_df")

    if calendar_df["d"].duplicated().any():
        raise ValueError("calendar_df contains duplicate 'd' keys")

    if calendar_df[list(CALENDAR_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("calendar_df contains missing required values")

    parsed_dates = pd.to_datetime(calendar_df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("calendar_df contains invalid dates")

    logger.info("calendar_df validation passed")


def validate_prices_data(prices_df: pd.DataFrame) -> None:
    _check_required_columns(prices_df, PRICES_REQUIRED_COLUMNS, "prices_df")

    if (prices_df["sell_price"] < 0).any():
        raise ValueError("prices_df contains negative sell_price values")

    key_columns = ["store_id", "item_id", "wm_yr_wk"]
    if prices_df[list(PRICES_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("prices_df contains missing required values")

    if prices_df.duplicated(subset=key_columns).any():
        raise ValueError("prices_df contains duplicate item-store-week rows")

    logger.info("prices_df validation passed")


def validate_raw_m5_data(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> None:
    validate_sales_data(sales_df)
    validate_calendar_data(calendar_df)
    validate_prices_data(prices_df)

    sales_days = {column for column in sales_df.columns if column.startswith("d_")}
    unknown_sales_days = sales_days - set(calendar_df["d"])
    if unknown_sales_days:
        raise ValueError(
            "sales_df contains day columns missing from calendar_df: "
            f"{sorted(unknown_sales_days)}"
        )

    logger.info("All raw dataset validations passed")
