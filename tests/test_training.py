import pandas as pd
import pytest

from app.training.datasets import build_train_valid_split
from app.training.evaluate import evaluate_predictions, smape


def test_build_train_valid_split() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store_id": ["CA_1"] * 60,
            "item_id": ["ITEM_1"] * 60,
            "sales_lag_28": [1.0] * 60,
            "sales_roll_mean_7": [1.0] * 60,
            "sales_roll_mean_28": [1.0] * 60,
            "target": [2.0] * 60,
        }
    )

    split = build_train_valid_split(df)

    assert len(split.train_df) > 0
    assert len(split.valid_df) > 0
    assert "sales_lag_28" in split.feature_columns


def test_evaluate_predictions_returns_expected_metrics() -> None:
    metrics = evaluate_predictions(
        pd.Series([0.0, 2.0, 4.0]), pd.Series([0.0, 1.0, 5.0])
    )

    assert metrics["rmse"] == pytest.approx((2 / 3) ** 0.5)
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["bias"] == pytest.approx(0.0)
    assert smape(pd.Series([0.0]), pd.Series([0.0])) == 0.0


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [([], []), ([1.0], [1.0, 2.0]), ([1.0], [float("nan")])],
)
def test_evaluate_predictions_rejects_invalid_inputs(actual, predicted) -> None:
    with pytest.raises(ValueError):
        evaluate_predictions(pd.Series(actual), pd.Series(predicted))


def test_train_valid_split_rejects_short_history() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "sales_lag_28": [1.0],
            "target": [2.0],
        }
    )

    with pytest.raises(ValueError, match="split is empty"):
        build_train_valid_split(df)
