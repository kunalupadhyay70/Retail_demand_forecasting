from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_cached_feature_table, get_cached_model_bundle
from app.api.schemas.forecast_request import (
    BatchForecastRequest,
    ItemStoreForecastRequest,
)
from app.api.schemas.forecast_response import (
    BatchForecastResponse,
    ItemStoreForecastResponse,
)
from app.inference.batch_forecast import run_batch_forecast
from app.inference.predict import generate_item_store_forecast
from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.utils.common import utc_timestamp
from app.utils.database import insert_api_forecast_log

router = APIRouter()
logger = get_logger()


@router.post("/forecast/item-store", response_model=ItemStoreForecastResponse)
def forecast_item_store(
    request: ItemStoreForecastRequest,
    model_bundle=Depends(get_cached_model_bundle),
    feature_df=Depends(get_cached_feature_table),
) -> ItemStoreForecastResponse:
    try:
        result = generate_item_store_forecast(
            item_id=request.item_id,
            store_id=request.store_id,
            forecast_date=request.forecast_date,
            model_bundle=model_bundle,
            feature_df=feature_df,
        )

        insert_api_forecast_log(
            {
                "timestamp": utc_timestamp(),
                "item_id": result["item_id"],
                "store_id": result["store_id"],
                "feature_row_date": result["feature_row_date"],
                "target_date": result["target_date"],
                "forecast_horizon_days": result["forecast_horizon_days"],
                "predicted_sales": result["predicted_sales"],
                "model_name": result["model_name"],
            }
        )

        return ItemStoreForecastResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected item-store inference error")
        raise HTTPException(status_code=500, detail="Internal inference error")


@router.post("/forecast/batch", response_model=BatchForecastResponse)
def forecast_batch(
    request: BatchForecastRequest,
    model_bundle=Depends(get_cached_model_bundle),
    feature_df=Depends(get_cached_feature_table),
) -> BatchForecastResponse:
    try:
        output_path = run_batch_forecast(
            model_bundle=model_bundle,
            feature_df=feature_df,
            limit=request.limit,
        )
        settings = get_settings()
        try:
            public_output_path = str(output_path.relative_to(settings.root_dir))
        except ValueError:
            public_output_path = output_path.name
        return BatchForecastResponse(
            status="success",
            output_path=public_output_path,
            requested_limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Unexpected batch inference error")
        raise HTTPException(status_code=500, detail="Internal batch inference error")
