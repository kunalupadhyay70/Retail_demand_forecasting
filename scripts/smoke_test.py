from __future__ import annotations

import argparse
import json
import urllib.request


def request_json(url: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a running M5 API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--item-id", default="FOODS_1_001")
    parser.add_argument("--store-id", default="CA_1")
    args = parser.parse_args()

    health = request_json(f"{args.base_url}/health")
    model_info = request_json(f"{args.base_url}/model-info")
    forecast = request_json(
        f"{args.base_url}/forecast/item-store",
        {"item_id": args.item_id, "store_id": args.store_id},
    )

    print(
        json.dumps(
            {"health": health, "model": model_info["model_name"], "forecast": forecast},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
