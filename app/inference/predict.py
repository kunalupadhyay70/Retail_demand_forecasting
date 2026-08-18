from __future__ import annotations

from datetime import date

import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.inference.load_model import load_model_bundle
from app.inference.prepare_features import (
    extract_model_input,
    get_model_categorical_schema,
    get_latest_feature_row,
    load_feature_table_for_inference,
)

logger = get_logger()


def generate_item_store_forecast(
    item_id: str,
    store_id: str,
    forecast_date: str | date | None = None,
    model_bundle: dict | None = None,
    feature_df: pd.DataFrame | None = None,
) -> dict:
    settings = get_settings()

    if model_bundle is None:
        model_bundle = load_model_bundle()

    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    feature_columns = metadata["feature_columns"]

    if feature_df is None:
        feature_df = load_feature_table_for_inference()

    latest_row = get_latest_feature_row(
        feature_df=feature_df,
        item_id=item_id,
        store_id=store_id,
        forecast_date=forecast_date,
    )

    X = extract_model_input(
        latest_row,
        feature_columns,
        categorical_schema=get_model_categorical_schema(model_bundle),
    )
    prediction = max(0.0, float(model.predict(X)[0]))

    row_date = pd.to_datetime(latest_row.loc[0, settings.feature_date_column])
    predicted_target_date = row_date + pd.Timedelta(days=settings.forecast_horizon_days)

    result = {
        "item_id": item_id,
        "store_id": store_id,
        "feature_row_date": row_date.strftime("%Y-%m-%d"),
        "target_date": predicted_target_date.strftime("%Y-%m-%d"),
        "forecast_horizon_days": settings.forecast_horizon_days,
        "predicted_sales": prediction,
        "model_name": metadata.get("model_name", "lightgbm_model"),
        "metrics": metadata.get("metrics", {}),
    }

    logger.info(
        "Generated prediction | item_id={} | store_id={} | predicted_sales={:.4f}",
        item_id,
        store_id,
        prediction,
    )

    return result
