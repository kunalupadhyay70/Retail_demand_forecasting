from __future__ import annotations

import pandas as pd

import app.monitoring.data_quality as data_quality
import app.monitoring.forecast_metrics as forecast_metrics
from app.monitoring.alerts import check_metric_alerts


def test_metric_alert_thresholds() -> None:
    alerts = check_metric_alerts({"rmse": 3.1, "mae": 1.0, "bias": -0.75})

    assert alerts == {
        "rmse_alert": True,
        "mae_alert": False,
        "bias_alert": True,
    }


def test_data_quality_counts_rows_missing_and_duplicates(monkeypatch) -> None:
    inserted: list[dict] = []
    monkeypatch.setattr(data_quality, "insert_data_quality_run", inserted.append)
    frame = pd.DataFrame({"a": [1, 1], "b": [None, None]})

    result = data_quality.run_data_quality_checks(frame, "test")

    assert result["row_count"] == 2
    assert result["missing_value_count"] == 2
    assert result["duplicate_count"] == 1
    assert inserted == [result]


def test_forecast_monitoring_joins_predictions_to_actuals(
    tmp_path, monkeypatch
) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "item_id": ["ITEM_1"],
            "store_id": ["CA_1"],
            "target_date": ["2016-01-02"],
            "predicted_sales": [2.0],
        }
    ).to_parquet(prediction_path, index=False)
    actuals = pd.DataFrame(
        {
            "item_id": ["ITEM_1"],
            "store_id": ["CA_1"],
            "date": pd.to_datetime(["2016-01-02"]),
            "sales": [3.0],
        }
    )
    inserted: list[dict] = []
    monkeypatch.setattr(forecast_metrics, "insert_monitoring_run", inserted.append)

    result = forecast_metrics.evaluate_prediction_file_against_actuals(
        prediction_path, actuals
    )

    assert result["joined_row_count"] == 1
    assert result["rmse"] == 1.0
    assert inserted == [result]
