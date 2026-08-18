from app.config.logging_config import configure_logging
from app.features.build_features import build_and_save_feature_table
from app.utils.paths import ensure_directories


def main() -> None:
    configure_logging()

    from app.config.logging_config import get_logger

    logger = get_logger()

    ensure_directories()

    logger.info("Starting feature engineering pipeline")
    output_path = build_and_save_feature_table()
    logger.info("Feature engineering pipeline completed successfully: {}", output_path)


if __name__ == "__main__":
    main()
