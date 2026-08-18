import app.orchestration.flows as flows
from app.orchestration.schedules import SUGGESTED_SCHEDULES


def test_suggested_schedules_present() -> None:
    assert "training_flow" in SUGGESTED_SCHEDULES
    assert "monitoring_flow" in SUGGESTED_SCHEDULES


def test_full_flow_includes_ingestion(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        flows, "initialize_environment_task", lambda: calls.append("init")
    )
    monkeypatch.setattr(
        flows,
        "ingest_raw_data_task",
        lambda: calls.append("ingest") or "base.parquet",
    )
    monkeypatch.setattr(
        flows,
        "build_features_task",
        lambda: calls.append("features") or "features.parquet",
    )
    monkeypatch.setattr(
        flows, "train_model_task", lambda: calls.append("train") or {"ok": True}
    )
    monkeypatch.setattr(
        flows,
        "run_batch_forecast_task",
        lambda limit: calls.append(f"batch:{limit}") or "predictions.parquet",
    )
    monkeypatch.setattr(
        flows,
        "run_monitoring_task",
        lambda: calls.append("monitor") or {"ok": True},
    )

    result = flows.full_pipeline_flow.fn(batch_limit=5)

    assert calls == ["init", "ingest", "features", "train", "batch:5", "monitor"]
    assert result["ingestion_output"] == "base.parquet"
