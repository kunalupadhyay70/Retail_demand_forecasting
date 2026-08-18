from fastapi import APIRouter, Depends

from app.api.dependencies import get_cached_model_bundle

router = APIRouter()


@router.get("/model-info")
def get_model_info(model_bundle=Depends(get_cached_model_bundle)) -> dict:
    return model_bundle["metadata"]
