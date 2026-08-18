from pathlib import Path

from app.config.settings import get_settings


def ensure_directories() -> None:
    settings = get_settings()

    directories = [
        settings.data_dir,
        settings.raw_data_dir,
        settings.interim_data_dir,
        settings.processed_data_dir,
        settings.predictions_dir,
        settings.artifacts_dir,
        settings.logs_dir,
        settings.mlruns_dir,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_raw_file_path(file_name: str) -> Path:
    settings = get_settings()
    return settings.raw_data_dir / file_name


def get_processed_file_path(file_name: str) -> Path:
    settings = get_settings()
    return settings.processed_data_dir / file_name


def get_artifact_path(file_name: str) -> Path:
    settings = get_settings()
    return settings.artifacts_dir / file_name
