from __future__ import annotations

import pytest

import app.utils.database as database
from app.config.settings import Settings


def test_database_initialization_insert_and_query(tmp_path, monkeypatch) -> None:
    settings = Settings(
        root_dir=tmp_path,
        database_url="sqlite:///artifacts/test.db",
        history_days_for_training=120,
    )
    monkeypatch.setattr(database, "get_settings", lambda: settings)

    database.initialize_database()
    database.insert_api_forecast_log(
        {
            "timestamp": "2026-01-01 00:00:00",
            "item_id": "ITEM_1",
            "store_id": "CA_1",
            "feature_row_date": "2016-03-27",
            "target_date": "2016-04-24",
            "forecast_horizon_days": 28,
            "predicted_sales": 1.25,
            "model_name": "test_model",
        }
    )

    rows = database.get_recent_api_forecast_logs(limit=1)

    assert database.get_database_file_path().exists()
    assert len(rows) == 1
    assert rows.iloc[0]["predicted_sales"] == pytest.approx(1.25)


@pytest.mark.parametrize("limit", [0, -1, True, "1"])
def test_database_rejects_unsafe_limits(limit) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        database._validate_limit(limit)


def test_database_rejects_non_sqlite_url(tmp_path, monkeypatch) -> None:
    settings = Settings(
        root_dir=tmp_path,
        database_url="postgresql://localhost/example",
        history_days_for_training=120,
    )
    monkeypatch.setattr(database, "get_settings", lambda: settings)

    with pytest.raises(ValueError, match="Unsupported database URL"):
        database.get_database_file_path()
