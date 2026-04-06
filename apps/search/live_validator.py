from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from apps.core.config import Settings
from apps.core.runtime_validation import RuntimeDependencyValidator
from apps.search.schema_registry import BRANCHES, PAYLOAD_INDEX_FIELDS, SearchSchemaRegistry

logger = logging.getLogger(__name__)


SAMPLE_SCAN_BATCH_SIZE = 32
SAMPLE_SCAN_LIMIT = 256
SAMPLE_COMPLETENESS_CHECKS = (
    "sample_root_fields",
    "sample_art_present",
    "sample_pjt_present",
    "sample_project_dates_valid",
)
OPTIONAL_CHECKS = {"sample_pat_present"}


@dataclass(slots=True)
class LiveValidationReport:
    ready: bool
    checks: dict[str, bool]
    issues: list[str]
    collection_name: str
    sample_point_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checks": self.checks,
            "issues": self.issues,
            "collection_name": self.collection_name,
            "sample_point_id": self.sample_point_id,
        }


class LiveContractValidator:
    def __init__(
        self,
        *,
        client: QdrantClient,
        settings: Settings,
        registry: SearchSchemaRegistry,
        dependency_validator: RuntimeDependencyValidator | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.registry = registry
        self.dependency_validator = dependency_validator or RuntimeDependencyValidator(settings)

    def _build_report(
        self,
        *,
        ready: bool,
        checks: dict[str, bool],
        issues: list[str],
        sample_point_id: str | None = None,
    ) -> LiveValidationReport:
        return LiveValidationReport(
            ready=ready,
            checks=checks,
            issues=issues,
            collection_name=self.settings.qdrant_collection_name,
            sample_point_id=sample_point_id,
        )

    @staticmethod
    def _is_nonempty_list(value: Any) -> bool:
        return isinstance(value, list) and bool(value)

    @staticmethod
    def _modifier_is_idf(modifier: Any) -> bool:
        if modifier is None:
            return False

        candidates = [modifier, getattr(modifier, "value", None), getattr(modifier, "name", None)]
        for candidate in candidates:
            if candidate is None:
                continue
            normalized = str(candidate).strip().lower()
            if normalized == "idf" or "modifier.idf" in normalized:
                return True
        return False

    def _build_sample_checks(self, sample_payload: dict[str, Any]) -> dict[str, bool]:
        required_root = {"basic_info", "researcher_profile"}
        art_items = sample_payload.get("publications")
        pat_items = sample_payload.get("intellectual_properties")
        pjt_items = sample_payload.get("research_projects")

        return {
            "sample_root_fields": required_root.issubset(sample_payload.keys()),
            "sample_art_present": self._is_nonempty_list(art_items),
            "sample_pat_present": self._is_nonempty_list(pat_items),
            "sample_pjt_present": self._is_nonempty_list(pjt_items),
            "sample_project_dates_valid": bool(
                isinstance(pjt_items, list)
                and pjt_items
                and all(
                    isinstance(item, dict)
                    and all(
                        key in item and item.get(key) not in (None, "")
                        for key in ("project_start_date", "project_end_date", "reference_year")
                    )
                    for item in pjt_items
                )
            ),
        }

    def _sample_completeness_score(self, sample_payload: Any) -> int:
        if not isinstance(sample_payload, dict):
            return -1
        checks = self._build_sample_checks(sample_payload)
        return sum(checks[name] for name in SAMPLE_COMPLETENESS_CHECKS)

    def _select_sample_point(self) -> Any | None:
        offset: Any | None = None
        scanned = 0
        best_sample: Any | None = None
        best_score = -2

        while scanned < SAMPLE_SCAN_LIMIT:
            limit = min(SAMPLE_SCAN_BATCH_SIZE, SAMPLE_SCAN_LIMIT - scanned)
            records, next_offset = self.client.scroll(
                collection_name=self.settings.qdrant_collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break

            scanned += len(records)
            for record in records:
                score = self._sample_completeness_score(getattr(record, "payload", None))
                if best_sample is None or score > best_score:
                    best_sample = record
                    best_score = score
                if score == len(SAMPLE_COMPLETENESS_CHECKS):
                    return record

            if next_offset is None or len(records) < limit:
                break
            offset = next_offset

        return best_sample

    def validate(self) -> LiveValidationReport:
        checks: dict[str, bool] = {}
        issues: list[str] = []
        sample_point_id: str | None = None

        backend_results = self.dependency_validator.validate_backends()
        checks["llm_backend_connected"] = all(item.ok for item in backend_results if item.name == "llm_backend")
        checks["embedding_backend_connected"] = all(item.ok for item in backend_results if item.name == "embedding_backend")
        for item in backend_results:
            if not item.ok:
                issues.append(item.detail)

        try:
            collection_info = self.client.get_collection(self.settings.qdrant_collection_name)
            checks["collection_exists"] = True
        except Exception as exc:
            checks["collection_exists"] = False
            issues.append(f"Qdrant collection lookup failed: {exc}")
            logger.warning("Qdrant collection lookup failed during readiness validation", exc_info=True)
            return self._build_report(ready=False, checks=checks, issues=issues)

        vector_config = getattr(collection_info.config.params, "vectors", None) or {}
        sparse_config = getattr(collection_info.config.params, "sparse_vectors", None) or {}
        payload_schema = getattr(collection_info, "payload_schema", {}) or {}

        dense_names = set(vector_config.keys() if isinstance(vector_config, dict) else [])
        sparse_names = set(sparse_config.keys() if isinstance(sparse_config, dict) else [])

        checks["dense_vectors_present"] = all(
            self.registry.dense_vector_by_branch[branch] in dense_names for branch in BRANCHES
        )
        if not checks["dense_vectors_present"]:
            issues.append("Required dense named vectors are missing.")

        checks["sparse_vectors_present"] = all(
            self.registry.sparse_vector_by_branch[branch] in sparse_names for branch in BRANCHES
        )
        if not checks["sparse_vectors_present"]:
            issues.append("Required sparse named vectors are missing.")

        checks["sparse_vectors_idf"] = True
        for branch in BRANCHES:
            vector_name = self.registry.sparse_vector_by_branch[branch]
            params = sparse_config.get(vector_name) if isinstance(sparse_config, dict) else None
            modifier = getattr(params, "modifier", None)
            if not self._modifier_is_idf(modifier):
                checks["sparse_vectors_idf"] = False
                issues.append(f"{vector_name} sparse vector modifier is not IDF.")
                break

        expected_index_keys = {field_name for field_name, _ in PAYLOAD_INDEX_FIELDS}
        available_index_keys = set(payload_schema.keys())
        checks["payload_indexes_present"] = expected_index_keys.issubset(available_index_keys)
        if not checks["payload_indexes_present"]:
            missing = sorted(expected_index_keys - available_index_keys)
            issues.append(f"Required payload indexes are missing: {missing}")

        try:
            sample = self._select_sample_point()
        except Exception as exc:
            checks["sample_point_exists"] = False
            issues.append(f"Sample point lookup failed: {exc}")
            logger.warning("Sample point lookup failed during readiness validation", exc_info=True)
            return self._build_report(ready=False, checks=checks, issues=issues)

        checks["sample_point_exists"] = sample is not None
        if sample is None:
            issues.append("No sample point is available in the collection.")
            return self._build_report(ready=False, checks=checks, issues=issues)

        sample_point_id = str(getattr(sample, "id", None))
        checks["sample_payload_valid"] = True

        try:
            sample_payload = sample.payload or {}
            if not isinstance(sample_payload, dict):
                raise TypeError("sample payload must be a JSON object")

            sample_checks = self._build_sample_checks(sample_payload)
            checks.update(sample_checks)

            if not checks["sample_root_fields"]:
                issues.append("Sample point is missing required root fields.")
            if not checks["sample_art_present"]:
                issues.append("Sample point is missing publications[].")
            if not checks["sample_pjt_present"]:
                issues.append("Sample point is missing research_projects[].")

            if not checks["sample_project_dates_valid"]:
                issues.append(
                    "Sample point research_projects[] entries must include "
                    "project_start_date, project_end_date, and reference_year."
                )
        except Exception as exc:
            checks["sample_payload_valid"] = False
            issues.append(f"Sample point payload inspection failed: {exc}")
            logger.warning("Sample payload inspection failed during readiness validation", exc_info=True)
            return self._build_report(
                ready=False,
                checks=checks,
                issues=issues,
                sample_point_id=sample_point_id,
            )

        return self._build_report(
            ready=all(value for key, value in checks.items() if key not in OPTIONAL_CHECKS),
            checks=checks,
            issues=issues,
            sample_point_id=sample_point_id,
        )
