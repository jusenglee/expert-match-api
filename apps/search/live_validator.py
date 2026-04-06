"""
Qdrant 컬렉션의 데이터 무결성과 시스템 의존성(LLM, Embedding)의 연결 상태를 실시간으로 점검하는 모듈입니다.
단순한 서버 연결 확인을 넘어, 실제 데이터의 스키마와 필드 존재 여부까지 샘플링을 통해 검증합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from apps.core.config import Settings
from apps.core.runtime_validation import RuntimeDependencyValidator
from apps.search.schema_registry import BRANCHES, PAYLOAD_INDEX_FIELDS, SearchSchemaRegistry

logger = logging.getLogger(__name__)


# 샘플 데이터 점검을 위한 설정
SAMPLE_SCAN_BATCH_SIZE = 32
SAMPLE_SCAN_LIMIT = 256
# 필수적인 데이터 완성도 체크 항목
SAMPLE_COMPLETENESS_CHECKS = (
    "sample_root_fields",
    "sample_art_present",
    "sample_pjt_present",
    "sample_project_dates_valid",
)
# 있으면 좋지만 없어도 전체 'Ready' 상태에는 영향을 주지 않는 항목
OPTIONAL_CHECKS = {"sample_pat_present"}


@dataclass(slots=True)
class LiveValidationReport:
    """검증 결과를 상세히 담고 있는 보고서 클래스입니다."""
    ready: bool                        # 시스템이 즉시 사용 가능한 상태인지 여부
    checks: dict[str, bool]            # 각 개별 점검 항목별 통과 여부
    issues: list[str]                  # 발견된 문제점들의 목록
    collection_name: str               # 점검 대상 컬렉션 이름
    sample_point_id: str | None = None # 점검에 사용된 대표 샘플의 ID

    def to_dict(self) -> dict[str, Any]:
        """보고서 내용을 딕셔너리 형태로 변환합니다."""
        return {
            "ready": self.ready,
            "checks": self.checks,
            "issues": self.issues,
            "collection_name": self.collection_name,
            "sample_point_id": self.sample_point_id,
        }


class LiveContractValidator:
    """
    코드가 기대하는 데이터 규약(Contract)이 실제 DB 및 서버 환경과 일치하는지 검증합니다.
    """
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
        """보고서 객체를 생성합니다."""
        return LiveValidationReport(
            ready=ready,
            checks=checks,
            issues=issues,
            collection_name=self.settings.qdrant_collection_name,
            sample_point_id=sample_point_id,
        )

    @staticmethod
    def _is_nonempty_list(value: Any) -> bool:
        """값이 비어있지 않은 리스트인지 확인합니다."""
        return isinstance(value, list) and bool(value)

    @staticmethod
    def _modifier_is_idf(modifier: Any) -> bool:
        """Qdrant의 스파스 벡터 수정자가 전형적인 IDF 설정인지 확인합니다."""
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
        """특정 샘플 포인트의 페이로드가 내부 데이터 규약을 준수하는지 점검합니다."""
        required_root = {"basic_info", "researcher_profile"}
        art_items = sample_payload.get("publications")
        pat_items = sample_payload.get("intellectual_properties")
        pjt_items = sample_payload.get("research_projects")

        return {
            # 필수 루트 필드 존재 여부
            "sample_root_fields": required_root.issubset(sample_payload.keys()),
            # 주요 실적 데이터 존재 여부
            "sample_art_present": self._is_nonempty_list(art_items),
            "sample_pat_present": self._is_nonempty_list(pat_items),
            "sample_pjt_present": self._is_nonempty_list(pjt_items),
            # 과제의 기간 및 연도 데이터 정합성
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
        """샘플의 데이터 완성도 점수를 계산합니다 (최고점 = 모든 필수 체크 통과)."""
        if not isinstance(sample_payload, dict):
            return -1
        checks = self._build_sample_checks(sample_payload)
        return sum(checks[name] for name in SAMPLE_COMPLETENESS_CHECKS)

    def _select_sample_point(self) -> Any | None:
        """검증에 적합한 가장 완성도 높은 샘플 데이터를 컬렉션에서 찾아 반환합니다."""
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
                # 완벽한 샘플을 찾으면 즉시 반환
                if score == len(SAMPLE_COMPLETENESS_CHECKS):
                    return record

            if next_offset is None or len(records) < limit:
                break
            offset = next_offset

        return best_sample

    def validate(self) -> LiveValidationReport:
        """전체 시스템 환경과 데이터 상태를 종합적으로 검증합니다."""
        checks: dict[str, bool] = {}
        issues: list[str] = []
        sample_point_id: str | None = None

        # 1. 백엔드 연결성 점검 (LLM, Embedding 서버)
        backend_results = self.dependency_validator.validate_backends()
        checks["llm_backend_connected"] = all(item.ok for item in backend_results if item.name == "llm_backend")
        checks["embedding_backend_connected"] = all(item.ok for item in backend_results if item.name == "embedding_backend")
        for item in backend_results:
            if not item.ok:
                issues.append(item.detail)

        # 2. Qdrant 컬렉션 존재 여부 확인
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

        # 3. Dense 벡터(의미 검색용) 설정 확인
        checks["dense_vectors_present"] = all(
            self.registry.dense_vector_by_branch[branch] in dense_names for branch in BRANCHES
        )
        if not checks["dense_vectors_present"]:
            issues.append("필수 Dense 벡터 구성이 컬렉션에 없습니다.")

        # 4. Sparse 벡터(키워드 검색용) 설정 및 IDF 수정자 확인
        checks["sparse_vectors_present"] = all(
            self.registry.sparse_vector_by_branch[branch] in sparse_names for branch in BRANCHES
        )
        if not checks["sparse_vectors_present"]:
            issues.append("필수 Sparse 벡터 구성이 컬렉션에 없습니다.")

        checks["sparse_vectors_idf"] = True
        for branch in BRANCHES:
            vector_name = self.registry.sparse_vector_by_branch[branch]
            params = sparse_config.get(vector_name) if isinstance(sparse_config, dict) else None
            modifier = getattr(params, "modifier", None)
            if not self._modifier_is_idf(modifier):
                checks["sparse_vectors_idf"] = False
                issues.append(f"{vector_name}의 Sparse 벡터 수정자가 IDF가 아닙니다.")
                break

        # 5. 페이로드 인덱스 존재 여부 확인
        expected_index_keys = {field_name for field_name, _ in PAYLOAD_INDEX_FIELDS}
        available_index_keys = set(payload_schema.keys())
        checks["payload_indexes_present"] = expected_index_keys.issubset(available_index_keys)
        if not checks["payload_indexes_present"]:
            missing = sorted(expected_index_keys - available_index_keys)
            issues.append(f"필수 페이로드 인덱스가 누락되었습니다: {missing}")

        # 6. 실제 데이터 샘플 샘플링 및 내용 검증
        try:
            sample = self._select_sample_point()
        except Exception as exc:
            checks["sample_point_exists"] = False
            issues.append(f"샘플 데이터 조회 실패: {exc}")
            logger.warning("Sample point lookup failed during readiness validation", exc_info=True)
            return self._build_report(ready=False, checks=checks, issues=issues)

        checks["sample_point_exists"] = sample is not None
        if sample is None:
            issues.append("컬렉션에 조회 가능한 샘플 데이터가 전혀 없습니다.")
            return self._build_report(ready=False, checks=checks, issues=issues)

        sample_point_id = str(getattr(sample, "id", None))
        checks["sample_payload_valid"] = True

        try:
            sample_payload = sample.payload or {}
            if not isinstance(sample_payload, dict):
                raise TypeError("샘플 페이로드가 JSON 객체 형식이 아닙니다.")

            # 샘플 내부의 세부 데이터 항목 점검
            sample_checks = self._build_sample_checks(sample_payload)
            checks.update(sample_checks)

            # 발견된 세부 문제점 로깅
            if not checks["sample_root_fields"]:
                issues.append("샘플 데이터에 필수 루트 필드(basic_info 등)가 누락되었습니다.")
            if not checks["sample_art_present"]:
                issues.append("샘플 데이터에 논문 실적(publications)이 없습니다.")
            if not checks["sample_pjt_present"]:
                issues.append("샘플 데이터에 과제 실적(research_projects)이 없습니다.")

            if not checks["sample_project_dates_valid"]:
                issues.append(
                    "과제 데이터(research_projects)에 시작/종료일 또는 기준 연도 정보가 누락되었습니다."
                )
        except Exception as exc:
            checks["sample_payload_valid"] = False
            issues.append(f"샘플 데이터 상세 분석 실패: {exc}")
            logger.warning("Sample payload inspection failed during readiness validation", exc_info=True)
            return self._build_report(
                ready=False,
                checks=checks,
                issues=issues,
                sample_point_id=sample_point_id,
            )

        # 전체 통과 여부 결정 (Optional 항목 제외한 모든 필수 항목이 True여야 함)
        return self._build_report(
            ready=all(value for key, value in checks.items() if key not in OPTIONAL_CHECKS),
            checks=checks,
            issues=issues,
            sample_point_id=sample_point_id,
        )
