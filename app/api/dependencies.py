from fastapi import Request

from app.config.settings import get_settings


def get_app_settings():
    return get_settings()


def get_cached_model_bundle(request: Request):
    return request.app.state.model_bundle


def get_cached_feature_table(request: Request):
    return request.app.state.feature_table
