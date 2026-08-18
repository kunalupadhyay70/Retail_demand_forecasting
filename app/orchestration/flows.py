from __future__ import annotations

from pathlib import Path

from prefect import flow, task

from app.config.logging_config import configure_logging, get_logger
from app.data.ingest import load_raw_m5_data
from app.data.storage import save_parquet
from app.data.transform import build_base_table
from app.data.validate import validate_raw_m5_data
from app.features.build_features import build_and_save_feature_table
from app.inference.batch_forecast import run_batch_forecast
from app.inference.load_model import load_model_bundle
from app.inference.prepare_features import load_feature_table_for_inference
from app.monitoring.alerts import check_metric_alerts
from app.monitoring.data_quality import run_data_quality_checks
from app.monitoring.drift import build_drift_report
from app.monitoring.forecast_metrics import (
    evaluate_prediction_file_against_actuals,
    load_actuals_base_table,
    load_latest_prediction_file,
)
from app.training.datasets import build_train_valid_split, load_feature_table
from app.training.train_lightgbm import train_lightgbm_model
from app.utils.database import initialize_database
from app.utils.paths import ensure_directories, get_processed_file_path
from app.config.settings import get_settings

configure_logging()
logger = get_logger()


@task(name="initialize_environment")
def initialize_environment_task() -> None:
    ensure_directories()
    initialize_database()
    logger.info("Environment initialized")


@task(name="ingest_raw_data")
def ingest_raw_data_task() -> Path:
    settings = get_settings()

    sales_df, calendar_df, prices_df = load_raw_m5_data()
    validate_raw_m5_data(sales_df, calendar_df, prices_df)

    base_df = build_base_table(sales_df, calendar_df, prices_df)

    output_path = get_processed_file_path(settings.processed_base_table_name)
    save_parquet(base_df, output_path)

    logger.info("Ingestion task completed successfully | output={}", output_path)
    return output_path


@task(name="build_features")
def build_features_task() -> Path:
    output_path = build_and_save_feature_table()
    logger.info("Feature build task completed successfully | output={}", output_path)
    return output_path


@task(name="train_model")
def train_model_task() -> dict:
    feature_df = load_feature_table()
    split = build_train_valid_split(feature_df)
    result = train_lightgbm_model(split)

    payload = {
        "metrics": result.metrics,
        "artifact_paths": {k: str(v) for k, v in result.artifact_paths.items()},
        "best_baseline": result.baseline_results.iloc[0].to_dict(),
    }

    logger.info("Training task completed successfully | metrics={}", result.metrics)
    return payload


@task(name="run_batch_forecast")
def run_batch_forecast_task(limit: int | None = 100) -> str:
    model_bundle = load_model_bundle()
    feature_df = load_feature_table_for_inference()
    output_path = run_batch_forecast(
        model_bundle=model_bundle,
        feature_df=feature_df,
        limit=limit,
    )
    logger.info("Batch forecast task completed successfully | output={}", output_path)
    return str(output_path)


@task(name="run_monitoring")
def run_monitoring_task() -> dict:
    feature_df = load_feature_table_for_inference()
    actual_df = load_actuals_base_table()

    dq_result = run_data_quality_checks(feature_df, dataset_name="feature_table")
    prediction_file = load_latest_prediction_file()
    metrics_result = evaluate_prediction_file_against_actuals(
        prediction_file=prediction_file,
        actual_df=actual_df,
    )
    drift_report_path = build_drift_report(feature_df)
    alerts = check_metric_alerts(metrics_result)

    result = {
        "data_quality": dq_result,
        "forecast_metrics": metrics_result,
        "drift_report_path": str(drift_report_path),
        "alerts": alerts,
    }

    logger.info("Monitoring task completed successfully | {}", result)
    return result


@flow(name="m5-ingestion-flow")
def ingestion_flow() -> str:
    initialize_environment_task()
    output_path = ingest_raw_data_task()
    return str(output_path)


@flow(name="m5-feature-build-flow")
def feature_build_flow() -> str:
    initialize_environment_task()
    output_path = build_features_task()
    return str(output_path)


@flow(name="m5-training-flow")
def training_flow() -> dict:
    initialize_environment_task()
    result = train_model_task()
    return result


@flow(name="m5-batch-forecast-flow")
def batch_forecast_flow(limit: int | None = 100) -> str:
    initialize_environment_task()
    output_path = run_batch_forecast_task(limit=limit)
    return output_path


@flow(name="m5-monitoring-flow")
def monitoring_flow() -> dict:
    initialize_environment_task()
    result = run_monitoring_task()
    return result


@flow(name="m5-full-pipeline-flow")
def full_pipeline_flow(batch_limit: int | None = 100) -> dict:
    initialize_environment_task()

    ingestion_output = ingest_raw_data_task()
    feature_output = build_features_task()
    training_result = train_model_task()
    batch_output = run_batch_forecast_task(limit=batch_limit)
    monitoring_result = run_monitoring_task()

    result = {
        "ingestion_output": str(ingestion_output),
        "feature_output": str(feature_output),
        "training_result": training_result,
        "batch_output": batch_output,
        "monitoring_result": monitoring_result,
    }

    logger.info("Full pipeline flow completed successfully")
    return result
