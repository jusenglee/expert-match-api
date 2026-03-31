from __future__ import annotations

import json
import sys

from qdrant_client import QdrantClient

from apps.core.config import get_settings
from apps.core.runtime_validation import validate_runtime_settings
from apps.search.live_validator import LiveContractValidator
from apps.search.schema_registry import SearchSchemaRegistry


def main() -> int:
    settings = get_settings()
    validate_runtime_settings(settings)

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        cloud_inference=settings.qdrant_cloud_inference,
    )
    validator = LiveContractValidator(
        client=client,
        settings=settings,
        registry=SearchSchemaRegistry.default(),
    )
    report = validator.validate()
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
