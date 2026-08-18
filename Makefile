.PHONY: download sync format format-check lint lint-fix typecheck test quality ingest features train api batch monitor dashboard pipeline smoke docker-build flow-features flow-training flow-batch flow-monitor flow-full

download:
	uv run python -m scripts.download_m5_data

sync:
	uv sync --frozen

format:
	uv run black app scripts tests dashboard

format-check:
	uv run black --check app scripts tests dashboard

lint:
	uv run ruff check app scripts tests dashboard

lint-fix:
	uv run ruff check --fix app scripts tests dashboard

typecheck:
	uv run mypy app scripts dashboard

test:
	uv run pytest --cov=app --cov-report=term-missing --cov-fail-under=50 -q

quality: format-check lint typecheck test

ingest:
	uv run python -m scripts.run_ingestion

features:
	uv run python -m scripts.run_feature_build

train:
	uv run python -m scripts.run_training

api:
	uv run python -m scripts.run_api

batch:
	uv run python -m scripts.run_batch_forecast

monitor:
	uv run python -m scripts.run_monitoring

dashboard:
	uv run python -m scripts.run_dashboard

pipeline: ingest features train batch monitor

smoke:
	uv run python -m scripts.smoke_test

docker-build:
	docker compose build

flow-features:
	uv run python -m scripts.run_flow --flow features

flow-training:
	uv run python -m scripts.run_flow --flow training

flow-batch:
	uv run python -m scripts.run_flow --flow batch --batch-limit 100

flow-monitor:
	uv run python -m scripts.run_flow --flow monitoring

flow-full:
	uv run python -m scripts.run_flow --flow full --batch-limit 100
