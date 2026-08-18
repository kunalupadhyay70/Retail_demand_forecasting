import pandas as pd

from app.data.transform import select_sales_history_window
from app.features.calendar_features import add_calendar_features
from app.features.lag_features import add_lag_features
from app.features.price_features import add_price_features
from app.inference.prepare_features import align_categorical_features


def test_inference_restores_all_null_categorical_columns() -> None:
    model_input = pd.DataFrame(
        {
            "item_id": ["ITEM_1"],
            "event_name_2": [None],
            "sales_lag_28": [1.0],
        }
    )

    out = align_categorical_features(
        model_input,
        {"item_id": ["ITEM_1", "ITEM_2"], "event_name_2": []},
    )

    assert isinstance(out["item_id"].dtype, pd.CategoricalDtype)
    assert isinstance(out["event_name_2"].dtype, pd.CategoricalDtype)
    assert list(out["item_id"].cat.categories) == ["ITEM_1", "ITEM_2"]
    assert list(out["event_name_2"].cat.categories) == []


def test_sales_history_is_filtered_before_melt() -> None:
    sales_df = pd.DataFrame(
        {
            "id": ["ITEM_1_CA_1"],
            "item_id": ["ITEM_1"],
            "dept_id": ["DEPT_1"],
            "cat_id": ["CAT_1"],
            "store_id": ["CA_1"],
            "state_id": ["CA"],
            "d_1": [1],
            "d_2": [2],
            "d_3": [3],
            "d_4": [4],
            "d_5": [5],
        }
    )
    calendar_df = pd.DataFrame(
        {
            "d": ["d_1", "d_2", "d_3", "d_4", "d_5"],
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        }
    )

    out = select_sales_history_window(sales_df, calendar_df, history_days=2)

    assert list(out.columns[-2:]) == ["d_4", "d_5"]


def test_calendar_features_add_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2016-01-01", "2016-01-02"]),
            "event_name_1": [None, "Holiday"],
            "event_type_1": [None, "Cultural"],
            "snap_CA": [0, 1],
            "snap_TX": [0, 0],
            "snap_WI": [1, 1],
        }
    )

    out = add_calendar_features(df)

    assert "day_of_week" in out.columns
    assert "is_weekend" in out.columns
    assert "has_event" in out.columns


def test_lag_features_create_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "store_id": ["CA_1"] * 40,
            "item_id": ["ITEM_1"] * 40,
            "date": pd.date_range("2016-01-01", periods=40, freq="D"),
            "sales": list(range(40)),
        }
    )

    out = add_lag_features(df)

    assert "sales_lag_1" in out.columns
    assert "sales_lag_7" in out.columns
    assert "sales_roll_mean_7" in out.columns


def test_price_features_create_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "store_id": ["CA_1"] * 40,
            "item_id": ["ITEM_1"] * 40,
            "date": pd.date_range("2016-01-01", periods=40, freq="D"),
            "sell_price": [10.0] * 40,
        }
    )

    out = add_price_features(df)

    assert "price_lag_1" in out.columns
    assert "price_change_1" in out.columns
    assert "price_roll_mean_7" in out.columns


def test_price_features_do_not_backfill_from_the_future() -> None:
    df = pd.DataFrame(
        {
            "store_id": ["CA_1"] * 4,
            "item_id": ["ITEM_1"] * 4,
            "date": pd.date_range("2016-01-01", periods=4, freq="D"),
            "sell_price": [None, None, 10.0, 10.0],
        }
    )

    out = add_price_features(df)

    assert pd.isna(out.loc[0, "sell_price_filled"])
    assert pd.isna(out.loc[1, "sell_price_filled"])
    assert out.loc[2, "sell_price_filled"] == 10.0
