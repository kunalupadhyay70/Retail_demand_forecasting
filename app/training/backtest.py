from __future__ import annotations

import pandas as pd

from app.config.logging_config import get_logger
from app.training.evaluate import evaluate_predictions

logger = get_logger()


def evaluate_multiple_prediction_sets(
    y_true: pd.Series,
    prediction_dict: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_name, preds in prediction_dict.items():
        metrics = evaluate_predictions(y_true=y_true, y_pred=preds)
        row: dict[str, object] = {**metrics, "model_name": model_name}
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    logger.info("Evaluated {} prediction sets", len(result_df))
    return result_df
