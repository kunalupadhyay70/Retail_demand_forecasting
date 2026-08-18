from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class ItemStoreForecastRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    item_id: str = Field(
        ..., min_length=1, max_length=64, description="Item identifier from M5"
    )
    store_id: str = Field(
        ..., min_length=1, max_length=64, description="Store identifier from M5"
    )
    forecast_date: date | None = Field(
        default=None,
        description=(
            "Optional target date (YYYY-MM-DD). It must be exactly one configured "
            "forecast horizon after an available feature row."
        ),
    )


class BatchForecastRequest(BaseModel):
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Optional number of item-store pairs to forecast in batch mode",
    )
