from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from app.orchestration.flows import (
    batch_forecast_flow,
    feature_build_flow,
    full_pipeline_flow,
    ingestion_flow,
    monitoring_flow,
    training_flow,
)

FLOW_REGISTRY: dict[str, Callable[..., Any]] = {
    "ingestion": ingestion_flow,
    "features": feature_build_flow,
    "training": training_flow,
    "batch": batch_forecast_flow,
    "monitoring": monitoring_flow,
    "full": full_pipeline_flow,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Prefect flows for the M5 forecasting platform"
    )
    parser.add_argument(
        "--flow",
        required=True,
        choices=FLOW_REGISTRY.keys(),
        help="Which flow to run",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=100,
        help="Optional batch limit for batch/full flows",
    )

    args = parser.parse_args()

    if args.flow == "batch":
        result = FLOW_REGISTRY[args.flow](limit=args.batch_limit)
    elif args.flow == "full":
        result = FLOW_REGISTRY[args.flow](batch_limit=args.batch_limit)
    else:
        result = FLOW_REGISTRY[args.flow]()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
