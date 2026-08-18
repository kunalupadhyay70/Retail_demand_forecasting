from pydantic import BaseModel


class ForecastMetrics(BaseModel):
    rmse: float | None = None
    mae: float | None = None
    smape: float | None = None
    bias: float | None = None


class ItemStoreForecastResponse(BaseModel):
    item_id: str
    store_id: str
    feature_row_date: str
    target_date: str
    forecast_horizon_days: int
    predicted_sales: float
    model_name: str
    metrics: ForecastMetrics


class BatchForecastResponse(BaseModel):
    status: str
    output_path: str
    requested_limit: int
