from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from app.config.logging_config import get_logger
from app.config.settings import get_settings

logger = get_logger()


def get_database_file_path() -> Path:
    settings = get_settings()
    db_url = settings.database_url

    if db_url.startswith("sqlite:///"):
        relative_path = db_url.replace("sqlite:///", "", 1)
        return settings.root_dir / relative_path

    raise ValueError(f"Unsupported database URL for local SQLite helper: {db_url}")


def get_connection() -> sqlite3.Connection:
    db_path = get_database_file_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_database() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_forecast_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            item_id TEXT NOT NULL,
            store_id TEXT NOT NULL,
            feature_row_date TEXT NOT NULL,
            target_date TEXT NOT NULL,
            forecast_horizon_days INTEGER NOT NULL,
            predicted_sales REAL NOT NULL,
            model_name TEXT NOT NULL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_forecast_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT NOT NULL,
            output_path TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            model_name TEXT NOT NULL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT NOT NULL,
            prediction_file TEXT NOT NULL,
            joined_row_count INTEGER NOT NULL,
            rmse REAL,
            mae REAL,
            smape REAL,
            bias REAL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_quality_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            missing_value_count INTEGER NOT NULL,
            duplicate_count INTEGER NOT NULL
        )
        """)

    logger.info("Initialized SQLite database tables")


def insert_api_forecast_log(payload: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
        INSERT INTO api_forecast_logs (
            timestamp,
            item_id,
            store_id,
            feature_row_date,
            target_date,
            forecast_horizon_days,
            predicted_sales,
            model_name
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                payload["timestamp"],
                payload["item_id"],
                payload["store_id"],
                payload["feature_row_date"],
                payload["target_date"],
                payload["forecast_horizon_days"],
                payload["predicted_sales"],
                payload["model_name"],
            ),
        )


def insert_batch_forecast_run(payload: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
        INSERT INTO batch_forecast_runs (
            run_timestamp,
            output_path,
            row_count,
            model_name
        )
        VALUES (?, ?, ?, ?)
        """,
            (
                payload["run_timestamp"],
                payload["output_path"],
                payload["row_count"],
                payload["model_name"],
            ),
        )


def insert_monitoring_run(payload: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
        INSERT INTO monitoring_runs (
            run_timestamp,
            prediction_file,
            joined_row_count,
            rmse,
            mae,
            smape,
            bias
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                payload["run_timestamp"],
                payload["prediction_file"],
                payload["joined_row_count"],
                payload["rmse"],
                payload["mae"],
                payload["smape"],
                payload["bias"],
            ),
        )


def insert_data_quality_run(payload: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
        INSERT INTO data_quality_runs (
            run_timestamp,
            dataset_name,
            row_count,
            missing_value_count,
            duplicate_count
        )
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                payload["run_timestamp"],
                payload["dataset_name"],
                payload["row_count"],
                payload["missing_value_count"],
                payload["duplicate_count"],
            ),
        )


def query_table(sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def _validate_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    return limit


def get_recent_batch_runs(limit: int = 10) -> pd.DataFrame:
    return query_table(
        """
        SELECT *
        FROM batch_forecast_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (_validate_limit(limit),),
    )


def get_recent_monitoring_runs(limit: int = 10) -> pd.DataFrame:
    return query_table(
        """
        SELECT *
        FROM monitoring_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (_validate_limit(limit),),
    )


def get_recent_data_quality_runs(limit: int = 10) -> pd.DataFrame:
    return query_table(
        """
        SELECT *
        FROM data_quality_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (_validate_limit(limit),),
    )


def get_recent_api_forecast_logs(limit: int = 20) -> pd.DataFrame:
    return query_table(
        """
        SELECT *
        FROM api_forecast_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (_validate_limit(limit),),
    )
