from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config.logging_config import get_logger

logger = get_logger()


def save_parquet(df: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_name(f".{file_path.name}.tmp")
    try:
        df.to_parquet(temp_path, index=False)
        temp_path.replace(file_path)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info(f"Saved parquet file to {file_path}")


def load_parquet(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    logger.info(f"Loading parquet file from {file_path}")
    return pd.read_parquet(file_path)
