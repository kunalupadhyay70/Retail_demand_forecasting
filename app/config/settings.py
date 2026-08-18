from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "M5 Demand Forecasting Platform"
    environment: str = "dev"
    debug: bool = True

    root_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = Path()
    raw_data_dir: Path = Path()
    interim_data_dir: Path = Path()
    processed_data_dir: Path = Path()
    predictions_dir: Path = Path()
    artifacts_dir: Path = Path()
    logs_dir: Path = Path()
    mlruns_dir: Path = Path()

    database_url: str = "sqlite:///artifacts/metadata.db"
    mlflow_tracking_uri: str = ""
    log_level: str = "INFO"

    sales_file_name: str = "sales_train_validation.csv"
    calendar_file_name: str = "calendar.csv"
    prices_file_name: str = "sell_prices.csv"

    processed_base_table_name: str = "base_table.parquet"
    feature_table_name: str = "feature_table.parquet"
    feature_build_start_date: str | None = None
    history_days_for_training: int = Field(default=120, ge=85)
    forecast_horizon_days: int = Field(default=28, ge=1)

    feature_date_column: str = "date"
    target_column: str = "target"
    model_artifact_dir_name: str = "models"
    train_validation_days: int = Field(default=28, ge=1)
    mlflow_experiment_name: str = "m5_demand_forecasting"
    random_state: int = 42

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8001, ge=1, le=65535)
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = Field(default=8501, ge=1, le=65535)
    max_batch_forecast_rows: int = Field(default=1000, ge=1, le=30490)
    model_file_name: str = "lightgbm_model.joblib"
    model_metadata_file_name: str = "lightgbm_model_metadata.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context) -> None:
        if self.data_dir == Path():
            self.data_dir = self.root_dir / "data"
        if self.raw_data_dir == Path():
            self.raw_data_dir = self.data_dir / "raw"
        if self.interim_data_dir == Path():
            self.interim_data_dir = self.data_dir / "interim"
        if self.processed_data_dir == Path():
            self.processed_data_dir = self.data_dir / "processed"
        if self.predictions_dir == Path():
            self.predictions_dir = self.data_dir / "predictions"
        if self.artifacts_dir == Path():
            self.artifacts_dir = self.root_dir / "artifacts"
        if self.logs_dir == Path():
            self.logs_dir = self.root_dir / "logs"
        if self.mlruns_dir == Path():
            self.mlruns_dir = self.root_dir / "mlruns"

        if not self.mlflow_tracking_uri:
            self.mlflow_tracking_uri = self.mlruns_dir.resolve().as_uri()


@lru_cache
def get_settings() -> Settings:
    return Settings()
