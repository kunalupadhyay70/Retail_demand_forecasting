from app.config.logging_config import configure_logging
from app.training.datasets import build_train_valid_split, load_feature_table
from app.training.train_lightgbm import train_lightgbm_model
from app.utils.paths import ensure_directories


def main() -> None:
    configure_logging()

    from app.config.logging_config import get_logger

    logger = get_logger()

    ensure_directories()

    logger.info("Starting training pipeline")

    feature_df = load_feature_table()
    split = build_train_valid_split(feature_df)

    result = train_lightgbm_model(split)

    logger.info("Training completed successfully")
    logger.info("Final LightGBM metrics: {}", result.metrics)
    logger.info(
        "Baseline comparison:\n{}", result.baseline_results.to_string(index=False)
    )
    logger.info("Saved artifacts: {}", result.artifact_paths)


if __name__ == "__main__":
    main()
