from __future__ import annotations

import pandas as pd

GROUP_KEYS = ["store_id", "item_id"]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sales"] = df["sales"].astype("float32")
    grouped = df.groupby(GROUP_KEYS, observed=False)["sales"]

    lag_days = [1, 7, 14, 28]
    for lag in lag_days:
        df[f"sales_lag_{lag}"] = grouped.shift(lag).astype("float32")

    df["sales_roll_mean_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    ).astype("float32")

    df["sales_roll_mean_14"] = grouped.transform(
        lambda s: s.shift(1).rolling(14, min_periods=5).mean()
    ).astype("float32")

    df["sales_roll_mean_28"] = grouped.transform(
        lambda s: s.shift(1).rolling(28, min_periods=7).mean()
    ).astype("float32")

    df["sales_roll_std_7"] = (
        grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())
        .fillna(0)
        .astype("float32")
    )

    df["sales_roll_std_28"] = (
        grouped.transform(lambda s: s.shift(1).rolling(28, min_periods=7).std())
        .fillna(0)
        .astype("float32")
    )

    df["sales_roll_min_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).min()
    ).astype("float32")

    df["sales_roll_max_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).max()
    ).astype("float32")

    df["sales_trend_short_vs_long"] = (
        df["sales_roll_mean_7"] / (df["sales_roll_mean_28"] + 1e-6)
    ).astype("float32")

    df["sales_lag_7_vs_28"] = (df["sales_lag_7"] / (df["sales_lag_28"] + 1e-6)).astype(
        "float32"
    )

    return df
