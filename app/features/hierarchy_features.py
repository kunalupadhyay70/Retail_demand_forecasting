from __future__ import annotations

import pandas as pd


def _add_group_rolling_mean(
    df: pd.DataFrame,
    group_cols: list[str],
    feature_name: str,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    agg_df = (
        df.groupby(group_cols + ["date"], as_index=False, observed=False)["sales"]
        .sum()
        .sort_values(group_cols + ["date"])
    )

    agg_df[feature_name] = (
        agg_df.groupby(group_cols, observed=False)["sales"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
        .astype("float32")
    )

    feature_cols = group_cols + ["date", feature_name]
    df = df.merge(agg_df[feature_cols], on=group_cols + ["date"], how="left")
    return df


def add_hierarchy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    hierarchy_specs = [
        (["store_id"], "store_roll_mean_7", 7, 3),
        (["store_id"], "store_roll_mean_28", 28, 7),
        (["cat_id"], "cat_roll_mean_7", 7, 3),
        (["cat_id"], "cat_roll_mean_28", 28, 7),
        (["dept_id"], "dept_roll_mean_7", 7, 3),
        (["dept_id"], "dept_roll_mean_28", 28, 7),
        (["state_id"], "state_roll_mean_7", 7, 3),
        (["state_id"], "state_roll_mean_28", 28, 7),
    ]

    for group_cols, feature_name, window, min_periods in hierarchy_specs:
        df = _add_group_rolling_mean(
            df=df,
            group_cols=group_cols,
            feature_name=feature_name,
            window=window,
            min_periods=min_periods,
        )

    return df
