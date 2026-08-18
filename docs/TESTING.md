# Testing strategy

This project uses layered tests so failures can be isolated before the real M5 pipeline consumes several gigabytes of memory.

## Automated layers

| Layer | Coverage |
|---|---|
| Unit | Metrics, dates, validation rules, feature transformations, schema alignment, alerts, timestamps |
| Component | SQLite persistence, atomic Parquet/model writes, safe archive extraction, batch output bounds |
| API | Lifespan setup, health/model metadata, successful inference, validation, safe error responses |
| Orchestration | Full-flow stage ordering, including ingestion |
| Smoke | A running API's health, model bundle, and real item/store forecast |

Run the complete local quality gate:

```bash
make quality
```

Run only tests:

```bash
uv run pytest --cov=app --cov-report=term-missing --cov-fail-under=50 -q
```

The latest verified run passed 41 tests with 70.57% application coverage on Python 3.12. CI runs the same suite on Python 3.11 and 3.12.

## Real-data integration check

Automated unit tests use small deterministic frames. Before presenting or releasing the project, also run the public dataset path:

```bash
make download
make pipeline
```

Confirm that it creates:

- `data/processed/base_table.parquet`
- `data/processed/feature_table.parquet`
- `artifacts/models/lightgbm_model.joblib`
- `artifacts/models/lightgbm_model_metadata.json`
- At least one `data/predictions/batch_forecast_*.parquet`
- `artifacts/drift_report.html`

## Service smoke checks

Local processes:

```bash
make api
make smoke
```

Container stack:

```bash
docker compose up --build -d
docker compose ps
uv run python -m scripts.smoke_test
```

Expected Compose state: both `api` and `dashboard` report `healthy`. Also open the dashboard and verify that model metrics, batch history, monitoring metrics, API logs, prediction preview, and the drift-report download render.

## Failure-path checklist

Before a portfolio release, verify these cases remain covered:

- Missing or malformed raw data fails before processing.
- Duplicate item/store sales IDs and duplicate price keys are rejected.
- Future price values are not used to fill historical rows.
- Unknown item/store requests return HTTP 400.
- Malformed requests return HTTP 422.
- Unexpected inference failures return a generic HTTP 500 response.
- Negative raw model output is clipped to zero demand.
- Batch requests above the configured maximum are rejected.
- Repeated batch runs create distinct output files.
- Missing model or feature artifacts prevent API startup visibly.

## Performance profile

The verified default `HISTORY_DAYS_FOR_TRAINING=120` run observed approximately:

| Stage | Peak resident memory |
|---|---:|
| Ingestion | 3.8 GB |
| Feature engineering | 4.7 GB |
| Training | 3.1 GB |
| Monitoring | 3.0 GB |

These are observations from one Linux machine, not guarantees. Use an 8 GB-or-larger environment for the default profile and measure again before increasing the history window.
