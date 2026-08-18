from __future__ import annotations

import pandas as pd
import pytest

from app.data.validate import (
    validate_calendar_data,
    validate_prices_data,
    validate_raw_m5_data,
    validate_sales_data,
)


def valid_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.DataFrame(
        {
            "id": ["ITEM_1_CA_1"],
            "item_id": ["ITEM_1"],
            "dept_id": ["DEPT_1"],
            "cat_id": ["CAT_1"],
            "store_id": ["CA_1"],
            "state_id": ["CA"],
            "d_1": [1],
        }
    )
    calendar = pd.DataFrame(
        {
            "date": ["2016-01-01"],
            "wm_yr_wk": [11601],
            "d": ["d_1"],
            "weekday": ["Friday"],
            "month": [1],
            "year": [2016],
        }
    )
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1"],
            "item_id": ["ITEM_1"],
            "wm_yr_wk": [11601],
            "sell_price": [2.5],
        }
    )
    return sales, calendar, prices


def test_valid_raw_frames_pass() -> None:
    validate_raw_m5_data(*valid_frames())


def test_raw_validation_rejects_sales_day_missing_from_calendar() -> None:
    sales, calendar, prices = valid_frames()
    sales["d_2"] = 0

    with pytest.raises(ValueError, match="missing from calendar_df"):
        validate_raw_m5_data(sales, calendar, prices)


def test_sales_reject_duplicate_and_negative_values() -> None:
    sales, _, _ = valid_frames()
    duplicate = pd.concat([sales, sales], ignore_index=True)
    negative = sales.copy()
    negative["d_1"] = -1

    with pytest.raises(ValueError, match="duplicate ids"):
        validate_sales_data(duplicate)
    with pytest.raises(ValueError, match="negative sales"):
        validate_sales_data(negative)

    missing = sales.copy()
    missing.loc[0, "d_1"] = None
    with pytest.raises(ValueError, match="missing sales"):
        validate_sales_data(missing)


def test_calendar_rejects_invalid_dates() -> None:
    _, calendar, _ = valid_frames()
    calendar["date"] = "not-a-date"

    with pytest.raises(ValueError, match="invalid dates"):
        validate_calendar_data(calendar)


def test_prices_reject_duplicate_keys() -> None:
    _, _, prices = valid_frames()
    duplicate = pd.concat([prices, prices], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate item-store-week"):
        validate_prices_data(duplicate)
