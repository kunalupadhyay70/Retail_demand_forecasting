from __future__ import annotations

import json

import app.training.register_model as register_model


class SerializableModel:
    value = 42


def test_model_bundle_contains_reproducibility_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(register_model, "get_artifact_path", lambda _: tmp_path)

    paths = register_model.save_model_bundle(
        model=SerializableModel(),
        feature_columns=["item_id", "sales_lag_28"],
        metrics={"rmse": 1.25},
        categorical_schema={"item_id": ["ITEM_1"]},
        training_context={"train_rows": 100, "validation_rows": 20},
        model_name="test_model",
    )
    metadata = json.loads(paths["metadata_path"].read_text())

    assert paths["model_path"].exists()
    assert metadata["schema_version"] == 1
    assert metadata["feature_count"] == 2
    assert metadata["categorical_schema"] == {"item_id": ["ITEM_1"]}
    assert metadata["training"]["train_rows"] == 100
    assert not any(tmp_path.glob("*.tmp"))
