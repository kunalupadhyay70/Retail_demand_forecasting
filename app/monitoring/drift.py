from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from app.config.logging_config import get_logger
from app.config.settings import get_settings

logger = get_logger()


def build_drift_report(feature_df: pd.DataFrame) -> Path:
    settings = get_settings()

    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"])

    max_date = feature_df["date"].max()
    recent_cutoff = max_date - pd.Timedelta(days=28)
    reference_cutoff = recent_cutoff - pd.Timedelta(days=28)

    reference_df = feature_df[
        (feature_df["date"] > reference_cutoff) & (feature_df["date"] <= recent_cutoff)
    ].copy()

    current_df = feature_df[feature_df["date"] > recent_cutoff].copy()

    feature_subset = [
        col
        for col in [
            "sales_lag_1",
            "sales_lag_7",
            "sales_roll_mean_7",
            "sales_roll_mean_28",
            "sell_price_filled",
            "price_roll_mean_28",
            "store_roll_mean_28",
            "cat_roll_mean_28",
        ]
        if col in feature_df.columns
    ]

    if not feature_subset:
        raise ValueError("No valid columns available for drift reporting")

    report = Report(metrics=[DataDriftPreset()])
    eval_result = report.run(
        reference_data=reference_df[feature_subset],
        current_data=current_df[feature_subset],
    )

    output_path = settings.artifacts_dir / "drift_report.html"
    eval_result.save_html(str(output_path))

    logger.info("Saved drift report to {}", output_path)
    return output_path
