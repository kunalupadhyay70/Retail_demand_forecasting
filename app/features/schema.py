from __future__ import annotations

import pandas as pd

CATEGORICAL_FEATURE_COLUMNS = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]


def build_categorical_schema(
    df: pd.DataFrame, feature_columns: list[str]
) -> dict[str, list[str]]:
    """Capture the ordered category values required for inference."""
    schema: dict[str, list[str]] = {}
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column not in feature_columns:
            continue
        categorical = df[column].astype("category")
        schema[column] = [str(value) for value in categorical.cat.categories]
    return schema
