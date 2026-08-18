from __future__ import annotations

import json
from pathlib import Path

import joblib

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.utils.paths import get_artifact_path

logger = get_logger()


def get_model_directory() -> Path:
    settings = get_settings()
    model_dir = get_artifact_path(settings.model_artifact_dir_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_model_file_path() -> Path:
    settings = get_settings()
    return get_model_directory() / settings.model_file_name


def get_model_metadata_path() -> Path:
    settings = get_settings()
    return get_model_directory() / settings.model_metadata_file_name


def load_trained_model():
    model_path = get_model_file_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model file not found: {model_path}")

    logger.info("Loading trained model from {}", model_path)
    model = joblib.load(model_path)
    return model


def load_model_metadata() -> dict:
    metadata_path = get_model_metadata_path()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata file not found: {metadata_path}")

    logger.info("Loading model metadata from {}", metadata_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


def load_model_bundle() -> dict:
    model = load_trained_model()
    metadata = load_model_metadata()
    return {
        "model": model,
        "metadata": metadata,
    }
