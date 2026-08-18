from app.config.logging_config import configure_logging
from app.config.settings import get_settings
from app.data.ingest import load_raw_m5_data
from app.data.storage import save_parquet
from app.data.transform import build_base_table
from app.data.validate import validate_raw_m5_data
from app.utils.paths import ensure_directories, get_processed_file_path


def main() -> None:
    configure_logging()

    from app.config.logging_config import get_logger

    logger = get_logger()

    ensure_directories()
    settings = get_settings()

    logger.info("Starting M5 ingestion pipeline")

    sales_df, calendar_df, prices_df = load_raw_m5_data()
    validate_raw_m5_data(sales_df, calendar_df, prices_df)

    base_df = build_base_table(sales_df, calendar_df, prices_df)

    output_path = get_processed_file_path(settings.processed_base_table_name)
    save_parquet(base_df, output_path)

    logger.info("Ingestion pipeline completed successfully")


if __name__ == "__main__":
    main()
