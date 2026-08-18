from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.utils.paths import get_artifact_path
from app.utils.common import utc_timestamp

logger = get_logger()


def get_model_artifact_dir() -> Path:
    settings = get_settings()
    model_dir = get_artifact_path(settings.model_artifact_dir_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_model_bundle(
    model,
    feature_columns: list[str],
    metrics: dict[str, float],
    categorical_schema: dict[str, list[str]],
    training_context: dict[str, Any],
    model_name: str = "lightgbm_model",
) -> dict[str, Path]:
    model_dir = get_model_artifact_dir()

    model_path = model_dir / f"{model_name}.joblib"
    metadata_path = model_dir / f"{model_name}_metadata.json"

    model_temp_path = model_path.with_name(f".{model_path.name}.tmp")
    metadata_temp_path = metadata_path.with_name(f".{metadata_path.name}.tmp")

    metadata = {
        "schema_version": 1,
        "model_name": model_name,
        "trained_at_utc": utc_timestamp(),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "categorical_schema": categorical_schema,
        "metrics": metrics,
        "training": training_context,
    }

    try:
        joblib.dump(model, model_temp_path)
        with metadata_temp_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        model_temp_path.replace(model_path)
        metadata_temp_path.replace(metadata_path)
    finally:
        model_temp_path.unlink(missing_ok=True)
        metadata_temp_path.unlink(missing_ok=True)

    logger.info(
        "Saved model bundle | model_path={} | metadata_path={}",
        model_path,
        metadata_path,
    )

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
    }
