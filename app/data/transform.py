from __future__ import annotations

import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings

logger = get_logger()

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def select_sales_history_window(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    *,
    history_days: int,
    feature_build_start_date: str | None = None,
) -> pd.DataFrame:
    """Keep only sales days that can reach the configured feature table.

    The raw M5 sales file is wide. Melting all 1,913 day columns creates more
    than 58 million rows even though feature generation immediately discards
    all but the configured history window. Selecting the same window before
    the melt keeps ingestion bounded without changing the resulting features.
    """
    day_columns = [col for col in sales_df.columns if col.startswith("d_")]
    calendar_days = calendar_df.loc[
        calendar_df["d"].isin(day_columns), ["d", "date"]
    ].copy()
    calendar_days["date"] = pd.to_datetime(calendar_days["date"])

    if calendar_days.empty:
        raise ValueError("No sales day columns match calendar day keys")

    if feature_build_start_date:
        cutoff_date = pd.to_datetime(feature_build_start_date)
    else:
        cutoff_date = calendar_days["date"].max() - pd.Timedelta(days=history_days - 1)

    selected_days = set(calendar_days.loc[calendar_days["date"] >= cutoff_date, "d"])
    selected_day_columns = [col for col in day_columns if col in selected_days]

    if not selected_day_columns:
        raise ValueError(
            f"No sales day columns remain for history cutoff {cutoff_date.date()}"
        )

    logger.info(
        "Selected sales history before melt | cutoff={} | day_columns={} of {}",
        cutoff_date.date(),
        len(selected_day_columns),
        len(day_columns),
    )
    return sales_df.loc[:, ID_COLUMNS + selected_day_columns]


def melt_sales_to_long(sales_df: pd.DataFrame) -> pd.DataFrame:
    day_columns = [col for col in sales_df.columns if col.startswith("d_")]

    sales_long = sales_df.melt(
        id_vars=ID_COLUMNS,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    logger.info(f"Sales melted to long format | shape={sales_long.shape}")
    return sales_long


def preprocess_calendar(calendar_df: pd.DataFrame) -> pd.DataFrame:
    calendar_df = calendar_df.copy()
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])

    logger.info(f"Calendar preprocessed | shape={calendar_df.shape}")
    return calendar_df


def preprocess_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    prices_df = prices_df.copy()
    logger.info(f"Prices preprocessed | shape={prices_df.shape}")
    return prices_df


def build_base_table(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    settings = get_settings()
    calendar_df = preprocess_calendar(calendar_df)
    prices_df = preprocess_prices(prices_df)
    sales_df = select_sales_history_window(
        sales_df,
        calendar_df,
        history_days=settings.history_days_for_training,
        feature_build_start_date=settings.feature_build_start_date,
    )
    sales_long = melt_sales_to_long(sales_df)

    merged_df = sales_long.merge(
        calendar_df,
        on="d",
        how="left",
        validate="many_to_one",
    )

    merged_df = merged_df.merge(
        prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )

    merged_df = merged_df.sort_values(["store_id", "item_id", "date"]).reset_index(
        drop=True
    )

    logger.info(f"Base table built successfully | shape={merged_df.shape}")
    return merged_df
