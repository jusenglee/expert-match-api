"""Qdrant 컬렉션 데이터 무결성 + 백엔드(LLM/Embedding) 연결 상태를 점검하는 모듈 (flat 모델, v2.1).

시작 시 단일 벡터(vector_e5i/vector_splade) 구성·payload 인덱스·flat chunk 표본을 검증해
런타임 KeyError/검색 실패를 예방한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from apps.core.config import Settings
from apps.core.runtime_validation import RuntimeDependencyValidator
from apps.domain.chunk_view import normalize_doc_date
from apps.search.doc_types import DOC_TYPES
from apps.search.schema_registry import (
    DENSE_VECTOR_NAME,
    PAYLOAD_INDEX_FIELDS,
    SPARSE_VECTOR_NAME,
)
from apps.search.sparse_runtime import SparseRuntimeConfig, model_requires_idf_modifier

logger = logging.getLogger(__name__)

SAMPLE_SCAN_BATCH_SIZE = 32
SAMPLE_SCAN_LIMIT = 256

# flat chunk payload 표본 점검: 필수 항목은 전부 True여야 ready.
SAMPLE_COMPLETENESS_CHECKS = (
    "sample_root_fields",   # flat 공통 식별 필드 존재
    "sample_doc_type_valid",  # doc_type이 5종 중 하나
)
OPTIONAL_CHECKS = {"sample_doc_attrs_present", "sample_doc_date_present"}

_REQUIRED_ROOT_FIELDS = {"researcher_id", "doc_type", "chunk_id", "chunk_text"}


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
    """코드가 기대하는 flat 데이터 규약이 실제 DB/백엔드와 일치하는지 검증."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        settings: Settings,
        dependency_validator: RuntimeDependencyValidator | None = None,
        sparse_runtime: SparseRuntimeConfig | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.dependency_validator = dependency_validator or RuntimeDependencyValidator(settings)
        self.sparse_runtime = sparse_runtime

    def _build_report(
        self, *, ready: bool, checks: dict[str, bool], issues: list[str], sample_point_id: str | None = None
    ) -> LiveValidationReport:
        return LiveValidationReport(
            ready=ready,
            checks=checks,
            issues=issues,
            collection_name=self.settings.qdrant_collection_name,
            sample_point_id=sample_point_id,
        )

    @staticmethod
    def _modifier_is_idf(modifier: Any) -> bool:
        if modifier is None:
            return False
        for candidate in (modifier, getattr(modifier, "value", None), getattr(modifier, "name", None)):
            if candidate is None:
                continue
            normalized = str(candidate).strip().lower()
            if normalized == "idf" or "modifier.idf" in normalized:
                return True
        return False

    def _requires_idf_modifier(self) -> bool:
        if self.sparse_runtime is not None:
            return self.sparse_runtime.requires_idf_modifier
        return model_requires_idf_modifier(self.settings.sparse_model_name)

    def _modifier_matches_expected(self, modifier: Any) -> bool:
        if self._requires_idf_modifier():
            return self._modifier_is_idf(modifier)
        return modifier is None

    def _build_sample_checks(self, sample_payload: dict[str, Any]) -> dict[str, bool]:
        """flat chunk payload 표본 규약 점검."""
        doc_type = sample_payload.get("doc_type")
        doc_attrs = sample_payload.get("doc_attrs")
        return {
            "sample_root_fields": _REQUIRED_ROOT_FIELDS.issubset(sample_payload.keys()),
            "sample_doc_type_valid": doc_type in set(DOC_TYPES),
            "sample_doc_attrs_present": isinstance(doc_attrs, dict),
            "sample_doc_date_present": normalize_doc_date(sample_payload.get("doc_date")) is not None,
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

        # 1. 백엔드 연결성
        backend_results = self.dependency_validator.validate_backends()
        checks["llm_backend_connected"] = all(item.ok for item in backend_results if item.name == "llm_backend")
        checks["embedding_backend_connected"] = all(item.ok for item in backend_results if item.name == "embedding_backend")
        for item in backend_results:
            if not item.ok:
                issues.append(item.detail)

        # 2. 컬렉션 존재
        try:
            collection_info = self.client.get_collection(self.settings.qdrant_collection_name)
            checks["collection_exists"] = True
        except Exception as exc:
            checks["collection_exists"] = False
            issues.append(f"Qdrant 컬렉션 조회 실패: {exc}")
            logger.warning("Qdrant collection lookup failed during readiness validation", exc_info=True)
            return self._build_report(ready=False, checks=checks, issues=issues)

        vector_config = getattr(collection_info.config.params, "vectors", None) or {}
        sparse_config = getattr(collection_info.config.params, "sparse_vectors", None) or {}
        payload_schema = getattr(collection_info, "payload_schema", {}) or {}
        dense_names = set(vector_config.keys() if isinstance(vector_config, dict) else [])
        sparse_names = set(sparse_config.keys() if isinstance(sparse_config, dict) else [])

        # 3. 단일 dense/sparse 벡터 존재
        checks["dense_vectors_present"] = DENSE_VECTOR_NAME in dense_names
        if not checks["dense_vectors_present"]:
            issues.append(f"필수 Dense 벡터({DENSE_VECTOR_NAME})가 컬렉션에 없습니다. (있는 것: {sorted(dense_names)})")
        checks["sparse_vectors_present"] = SPARSE_VECTOR_NAME in sparse_names
        if not checks["sparse_vectors_present"]:
            issues.append(f"필수 Sparse 벡터({SPARSE_VECTOR_NAME})가 컬렉션에 없습니다. (있는 것: {sorted(sparse_names)})")

        # 4. Sparse modifier
        params = sparse_config.get(SPARSE_VECTOR_NAME) if isinstance(sparse_config, dict) else None
        modifier = getattr(params, "modifier", None)
        checks["sparse_vectors_idf"] = self._modifier_matches_expected(modifier)
        if not checks["sparse_vectors_idf"]:
            expected = "IDF" if self._requires_idf_modifier() else "None"
            issues.append(f"{SPARSE_VECTOR_NAME}의 Sparse 수정자가 기대값({expected})과 다릅니다.")

        # 5. payload 인덱스
        expected_index_keys = {field_name for field_name, _ in PAYLOAD_INDEX_FIELDS}
        available_index_keys = set(payload_schema.keys())
        checks["payload_indexes_present"] = expected_index_keys.issubset(available_index_keys)
        if not checks["payload_indexes_present"]:
            missing = sorted(expected_index_keys - available_index_keys)
            issues.append(f"필수 페이로드 인덱스 누락: {missing}")

        # 6. flat chunk 표본 검증
        try:
            sample = self._select_sample_point()
        except Exception as exc:
            checks["sample_point_exists"] = False
            issues.append(f"샘플 데이터 조회 실패: {exc}")
            logger.warning("Sample point lookup failed during readiness validation", exc_info=True)
            return self._build_report(ready=False, checks=checks, issues=issues)

        checks["sample_point_exists"] = sample is not None
        if sample is None:
            issues.append("컬렉션에 조회 가능한 샘플 데이터가 없습니다.")
            return self._build_report(ready=False, checks=checks, issues=issues)

        sample_point_id = str(getattr(sample, "id", None))
        checks["sample_payload_valid"] = True
        try:
            sample_payload = sample.payload or {}
            if not isinstance(sample_payload, dict):
                raise TypeError("샘플 페이로드가 JSON 객체 형식이 아닙니다.")
            sample_checks = self._build_sample_checks(sample_payload)
            checks.update(sample_checks)
            if not checks["sample_root_fields"]:
                issues.append("샘플 데이터에 flat 필수 루트 필드(researcher_id/doc_type/chunk_id/chunk_text)가 누락되었습니다.")
            if not checks["sample_doc_type_valid"]:
                issues.append(f"샘플 doc_type이 유효 5종이 아닙니다: {sample_payload.get('doc_type')!r}")
        except Exception as exc:
            checks["sample_payload_valid"] = False
            issues.append(f"샘플 데이터 상세 분석 실패: {exc}")
            logger.warning("Sample payload inspection failed during readiness validation", exc_info=True)
            return self._build_report(ready=False, checks=checks, issues=issues, sample_point_id=sample_point_id)

        return self._build_report(
            ready=all(value for key, value in checks.items() if key not in OPTIONAL_CHECKS),
            checks=checks,
            issues=issues,
            sample_point_id=sample_point_id,
        )
