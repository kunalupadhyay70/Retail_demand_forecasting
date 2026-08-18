from __future__ import annotations

import pandas as pd

GROUP_KEYS = ["store_id", "item_id"]


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sell_price" not in df.columns:
        raise ValueError("sell_price column is required for price feature generation")

    df["sell_price"] = df["sell_price"].astype("float32")
    df["sell_price_filled"] = (
        df.groupby(GROUP_KEYS, observed=False)["sell_price"]
        .transform(lambda s: s.ffill())
        .astype("float32")
    )

    grouped = df.groupby(GROUP_KEYS, observed=False)["sell_price_filled"]

    df["price_lag_1"] = grouped.shift(1).astype("float32")
    df["price_lag_7"] = grouped.shift(7).astype("float32")

    df["price_change_1"] = (df["sell_price_filled"] - df["price_lag_1"]).astype(
        "float32"
    )
    df["price_change_7"] = (df["sell_price_filled"] - df["price_lag_7"]).astype(
        "float32"
    )

    df["price_pct_change_1"] = (
        (df["sell_price_filled"] - df["price_lag_1"]) / (df["price_lag_1"] + 1e-6)
    ).astype("float32")

    df["price_pct_change_7"] = (
        (df["sell_price_filled"] - df["price_lag_7"]) / (df["price_lag_7"] + 1e-6)
    ).astype("float32")

    df["price_roll_mean_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    ).astype("float32")

    df["price_roll_std_7"] = (
        grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())
        .fillna(0)
        .astype("float32")
    )

    df["price_roll_mean_28"] = grouped.transform(
        lambda s: s.shift(1).rolling(28, min_periods=7).mean()
    ).astype("float32")

    df["price_vs_roll_mean_28"] = (
        df["sell_price_filled"] / (df["price_roll_mean_28"] + 1e-6)
    ).astype("float32")

    df["price_is_missing"] = df["sell_price"].isna().astype("int8")

    return df
