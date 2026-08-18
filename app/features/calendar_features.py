from __future__ import annotations

import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    df["day_of_month"] = df["date"].dt.day.astype("int8")
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["quarter"] = df["date"].dt.quarter.astype("int8")
    df["year"] = df["date"].dt.year.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_month_start"] = df["date"].dt.is_month_start.astype("int8")
    df["is_month_end"] = df["date"].dt.is_month_end.astype("int8")

    if "event_name_1" in df.columns:
        df["has_event"] = df["event_name_1"].notna().astype("int8")
    else:
        df["has_event"] = 0

    if "event_type_1" in df.columns:
        df["has_event_type"] = df["event_type_1"].notna().astype("int8")
    else:
        df["has_event_type"] = 0

    snap_columns = [
        col for col in ["snap_CA", "snap_TX", "snap_WI"] if col in df.columns
    ]
    for col in snap_columns:
        df[col] = df[col].fillna(0).astype("int8")

    return df
