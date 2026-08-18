from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.inference.batch_forecast as batch_forecast
from app.config.settings import Settings
from app.inference.batch_forecast import (
    build_latest_item_store_index,
    generate_batch_predictions_from_latest_rows,
    run_batch_forecast,
)
from app.inference.prepare_features import (
    get_latest_feature_row,
    get_model_categorical_schema,
)


class NegativeModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), -2.0)


def model_bundle() -> dict:
    return {
        "model": NegativeModel(),
        "metadata": {
            "model_name": "negative_model",
            "feature_columns": ["sales_lag_28"],
            "categorical_schema": {},
            "metrics": {"rmse": 1.0},
        },
    }


def feature_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["ITEM_1", "ITEM_1", "ITEM_2"],
            "store_id": ["CA_1", "CA_1", "CA_1"],
            "date": pd.to_datetime(["2016-01-01", "2016-01-02", "2016-01-02"]),
            "sales_lag_28": [1.0, 2.0, 3.0],
        }
    )


def test_latest_feature_selection_and_target_date() -> None:
    latest = get_latest_feature_row(feature_rows(), "ITEM_1", "CA_1")
    dated = get_latest_feature_row(
        feature_rows(), "ITEM_1", "CA_1", forecast_date="2016-01-30"
    )

    assert latest.loc[0, "date"] == pd.Timestamp("2016-01-02")
    assert dated.loc[0, "date"] == pd.Timestamp("2016-01-02")

    with pytest.raises(ValueError, match="No feature rows"):
        get_latest_feature_row(feature_rows(), "UNKNOWN", "CA_1")

    with pytest.raises(ValueError, match="No feature row can produce"):
        get_latest_feature_row(
            feature_rows(), "ITEM_1", "CA_1", forecast_date="2016-01-01"
        )


def test_batch_predictions_use_latest_rows_and_clip_negative_demand() -> None:
    latest = build_latest_item_store_index(feature_rows())
    result = generate_batch_predictions_from_latest_rows(latest, model_bundle())

    assert len(result) == 2
    assert (result["predicted_sales"] == 0).all()
    assert set(result["target_date"]) == {"2016-01-30"}


def test_batch_run_has_unique_atomic_outputs(tmp_path, monkeypatch) -> None:
    settings = Settings(
        root_dir=tmp_path,
        predictions_dir=tmp_path / "predictions",
        history_days_for_training=120,
        max_batch_forecast_rows=2,
    )
    logged_runs: list[dict] = []
    monkeypatch.setattr(batch_forecast, "get_settings", lambda: settings)
    monkeypatch.setattr(batch_forecast, "insert_batch_forecast_run", logged_runs.append)

    first = run_batch_forecast(model_bundle(), feature_rows(), limit=2)
    second = run_batch_forecast(model_bundle(), feature_rows(), limit=2)

    assert first.exists() and second.exists()
    assert first != second
    assert len(logged_runs) == 2

    with pytest.raises(ValueError, match="between 1 and 2"):
        run_batch_forecast(model_bundle(), feature_rows(), limit=3)


def test_categorical_schema_requires_valid_bundle() -> None:
    with pytest.raises(ValueError, match="valid categorical schema"):
        get_model_categorical_schema(
            {
                "model": type(
                    "Model",
                    (),
                    {"booster_": type("Booster", (), {"pandas_categorical": []})()},
                )(),
                "metadata": {"feature_columns": ["item_id"]},
            }
        )
