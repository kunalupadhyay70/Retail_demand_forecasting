from app.config.logging_config import configure_logging
from app.inference.prepare_features import load_feature_table_for_inference
from app.monitoring.alerts import check_metric_alerts
from app.monitoring.data_quality import run_data_quality_checks
from app.monitoring.drift import build_drift_report
from app.monitoring.forecast_metrics import (
    evaluate_prediction_file_against_actuals,
    load_actuals_base_table,
    load_latest_prediction_file,
)
from app.utils.database import initialize_database
from app.utils.paths import ensure_directories


def main() -> None:
    configure_logging()

    from app.config.logging_config import get_logger

    logger = get_logger()

    ensure_directories()
    initialize_database()

    logger.info("Starting monitoring pipeline")

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

    logger.info("Monitoring pipeline completed successfully")
    logger.info("Data quality result: {}", dq_result)
    logger.info("Forecast metrics result: {}", metrics_result)
    logger.info("Drift report: {}", drift_report_path)
    logger.info("Alerts: {}", alerts)


if __name__ == "__main__":
    main()
