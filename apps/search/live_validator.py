from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from apps.core.config import Settings
from apps.core.runtime_validation import RuntimeDependencyValidator
from apps.search.schema_registry import BRANCHES, PAYLOAD_INDEX_FIELDS, SearchSchemaRegistry


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
            issues.append(f"Qdrant 컬렉션 조회 실패: {exc}")
            return LiveValidationReport(
                ready=False,
                checks=checks,
                issues=issues,
                collection_name=self.settings.qdrant_collection_name,
            )

        vector_config = getattr(collection_info.config.params, "vectors", None) or {}
        sparse_config = getattr(collection_info.config.params, "sparse_vectors", None) or {}
        payload_schema = getattr(collection_info, "payload_schema", {}) or {}

        dense_names = set(vector_config.keys() if isinstance(vector_config, dict) else [])
        sparse_names = set(sparse_config.keys() if isinstance(sparse_config, dict) else [])

        checks["dense_vectors_present"] = all(
            self.registry.dense_vector_by_branch[branch] in dense_names for branch in BRANCHES
        )
        if not checks["dense_vectors_present"]:
            issues.append("필수 dense named vector가 일부 누락되었습니다.")

        checks["sparse_vectors_present"] = all(
            self.registry.sparse_vector_by_branch[branch] in sparse_names for branch in BRANCHES
        )
        if not checks["sparse_vectors_present"]:
            issues.append("필수 sparse named vector가 일부 누락되었습니다.")

        checks["sparse_vectors_idf"] = True
        for branch in BRANCHES:
            vector_name = self.registry.sparse_vector_by_branch[branch]
            params = sparse_config.get(vector_name) if isinstance(sparse_config, dict) else None
            modifier = getattr(params, "modifier", None)
            modifier_name = str(modifier).lower() if modifier is not None else ""
            if "idf" not in modifier_name:
                checks["sparse_vectors_idf"] = False
                issues.append(f"{vector_name} sparse vector modifier가 IDF가 아닙니다.")
                break

        expected_index_keys = {field_name for field_name, _ in PAYLOAD_INDEX_FIELDS}
        available_index_keys = set(payload_schema.keys())
        checks["payload_indexes_present"] = expected_index_keys.issubset(available_index_keys)
        if not checks["payload_indexes_present"]:
            missing = sorted(expected_index_keys - available_index_keys)
            issues.append(f"필수 payload index 누락: {missing}")

        try:
            records, _ = self.client.scroll(
                collection_name=self.settings.qdrant_collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            checks["sample_point_exists"] = False
            issues.append(f"샘플 point 조회 실패: {exc}")
            return LiveValidationReport(
                ready=False,
                checks=checks,
                issues=issues,
                collection_name=self.settings.qdrant_collection_name,
            )

        checks["sample_point_exists"] = bool(records)
        if not records:
            issues.append("실데이터 sample point가 없습니다.")
            return LiveValidationReport(
                ready=False,
                checks=checks,
                issues=issues,
                collection_name=self.settings.qdrant_collection_name,
            )

        sample = records[0]
        sample_payload = sample.payload or {}
        sample_point_id = str(getattr(sample, "id", None))

        required_root = {"doc_id", "blng_org_nm_exact"}
        checks["sample_root_fields"] = required_root.issubset(sample_payload.keys())
        if not checks["sample_root_fields"]:
            issues.append("sample point의 root 필수 필드가 부족합니다.")

        checks["sample_art_present"] = bool(sample_payload.get("art"))
        checks["sample_pat_present"] = bool(sample_payload.get("pat"))
        checks["sample_pjt_present"] = bool(sample_payload.get("pjt"))
        if not checks["sample_art_present"]:
            issues.append("sample point에 논문 근거 art[]가 없습니다.")
        if not checks["sample_pat_present"]:
            issues.append("sample point에 특허 근거 pat[]가 없습니다.")
        if not checks["sample_pjt_present"]:
            issues.append("sample point에 과제 근거 pjt[]가 없습니다.")

        pjt_items = sample_payload.get("pjt") or []
        checks["sample_project_dates_valid"] = bool(
            pjt_items
            and all(
                all(key in item and item.get(key) not in (None, "") for key in ("start_dt", "end_dt", "stan_yr"))
                for item in pjt_items
            )
        )
        if not checks["sample_project_dates_valid"]:
            issues.append("sample point의 pjt 날짜 필드(start_dt/end_dt/stan_yr)가 정합하지 않습니다.")

        ready = all(checks.values())
        return LiveValidationReport(
            ready=ready,
            checks=checks,
            issues=issues,
            collection_name=self.settings.qdrant_collection_name,
            sample_point_id=sample_point_id,
        )

