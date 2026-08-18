from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import forecast, health, model_info
from app.config.logging_config import configure_logging, get_logger
from app.config.settings import get_settings
from app.inference.load_model import load_model_bundle
from app.inference.prepare_features import load_feature_table_for_inference
from app.utils.database import initialize_database

configure_logging()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application state")
    initialize_database()

    app.state.model_bundle = load_model_bundle()
    app.state.feature_table = load_feature_table_for_inference()

    logger.info("Application state initialized successfully")
    yield
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.project_name,
        version="0.1.0",
        description="Production-style retail forecasting API using Walmart M5 data",
        lifespan=lifespan,
    )
    application.include_router(health.router, tags=["health"])
    application.include_router(model_info.router, tags=["model"])
    application.include_router(forecast.router, tags=["forecast"])
    return application


app = create_app()
