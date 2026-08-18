from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import app.api.main as api_main
import app.api.routes.forecast as forecast_routes


class FakeModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert list(frame.columns) == ["sales_lag_28"]
        return np.array([-0.25])


def test_api_lifespan_and_forecast_contract(monkeypatch) -> None:
    model_bundle = {
        "model": FakeModel(),
        "metadata": {
            "model_name": "test_model",
            "feature_columns": ["sales_lag_28"],
            "feature_count": 1,
            "categorical_schema": {},
            "metrics": {"rmse": 1.0, "mae": 0.5, "smape": 20.0, "bias": 0.0},
        },
    }
    feature_table = pd.DataFrame(
        {
            "item_id": ["ITEM_1"],
            "store_id": ["CA_1"],
            "date": pd.to_datetime(["2016-03-27"]),
            "sales_lag_28": [2.0],
        }
    )
    inserted_logs: list[dict] = []

    monkeypatch.setattr(api_main, "initialize_database", lambda: None)
    monkeypatch.setattr(api_main, "load_model_bundle", lambda: model_bundle)
    monkeypatch.setattr(
        api_main, "load_feature_table_for_inference", lambda: feature_table
    )
    monkeypatch.setattr(
        forecast_routes, "insert_api_forecast_log", inserted_logs.append
    )

    with TestClient(api_main.create_app()) as client:
        health = client.get("/health")
        model_info = client.get("/model-info")
        forecast = client.post(
            "/forecast/item-store",
            json={"item_id": "ITEM_1", "store_id": "CA_1"},
        )
        unknown_item = client.post(
            "/forecast/item-store",
            json={"item_id": "UNKNOWN", "store_id": "CA_1"},
        )
        oversized_batch = client.post("/forecast/batch", json={"limit": 1001})

    assert health.status_code == 200
    assert model_info.status_code == 200
    assert model_info.json()["model_name"] == "test_model"
    assert forecast.status_code == 200
    assert forecast.json()["predicted_sales"] == 0.0
    assert forecast.json()["target_date"] == "2016-04-24"
    assert inserted_logs[0]["item_id"] == "ITEM_1"
    assert unknown_item.status_code == 400
    assert oversized_batch.status_code == 422


def test_api_rejects_blank_item_id(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "initialize_database", lambda: None)
    monkeypatch.setattr(
        api_main,
        "load_model_bundle",
        lambda: {"model": FakeModel(), "metadata": {}},
    )
    monkeypatch.setattr(api_main, "load_feature_table_for_inference", pd.DataFrame)

    with TestClient(api_main.create_app()) as client:
        response = client.post(
            "/forecast/item-store", json={"item_id": " ", "store_id": "CA_1"}
        )

    assert response.status_code == 422
