from __future__ import annotations

from app.config.logging_config import get_logger

logger = get_logger()


def check_metric_alerts(metrics: dict) -> dict:
    alerts = {
        "rmse_alert": (
            metrics.get("rmse", 0) > 3.0 if metrics.get("rmse") is not None else False
        ),
        "mae_alert": (
            metrics.get("mae", 0) > 2.0 if metrics.get("mae") is not None else False
        ),
        "bias_alert": (
            abs(metrics.get("bias", 0)) > 0.5
            if metrics.get("bias") is not None
            else False
        ),
    }

    logger.info("Computed monitoring alerts: {}", alerts)
    return alerts
