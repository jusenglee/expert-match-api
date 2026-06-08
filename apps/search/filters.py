"""사용자 질의 분석 결과(hard_filters)를 Qdrant 필터(models.Filter)로 변환하는 모듈.

flat chunk 모델(v2.1) 기준. hard filter는 실데이터 샘플 계약상 안정적인 payload root
필드만 대상으로 삼는다. doc_type별 `doc_attrs.*`는 유동 필드이므로 필터 대상으로 쓰지 않는다.
DATA_CONTRACT §1.1 허용 키(`highest_degree` / `recent_years`+`recent_doc_types` /
`*_count_min`)를 다음 flat root 경로로 매핑한다:
- 연구자 공통 count/메타 → flat root (researcher_meta 중첩 폐기)
- recency → root `doc_date`(datetime), event_year 폐기

HARD 제약(보존 필수): 다중 doc_type recency는 AND가 아니라 **OR(min_should, min_count=1)**.
AND로 묶으면 0건 회귀(DATA_MODEL §3.2 교훈).
소속 기관 include/exclude는 정규화된 root 필드가 없으므로 Qdrant exact pre-filter가 아니라
retriever 앱단 post-filter에서 root `affiliated_organization`만 기준으로 처리한다.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from qdrant_client import models

from apps.search.doc_types import DOC_TYPE_TO_FAMILY, DOC_TYPES, Family

# 최소 건수 hard_filters 키 → flat root 백엔드 필드 (DATA_CONTRACT §1.1).
# 실데이터는 assessor를 단일 researcher_assessor_activity_count로 보유하므로, 구 split 키
# (researcher_assessor_count_min/expert_assessor_count_min)는 모두 그 단일 필드로 흡수한다(하위호환).
_COUNT_MIN_TO_FIELD: dict[str, str] = {
    "publication_count_min": "publication_count",
    "scie_publication_count_min": "scie_publication_count",
    "intellectual_property_count_min": "intellectual_property_count",
    "research_project_count_min": "research_project_count",
    "researcher_assessor_activity_count_min": "researcher_assessor_activity_count",
    # 하위호환 별칭(구 contract 키) → 단일 assessor count
    "assessor_activity_count_min": "researcher_assessor_activity_count",
    "researcher_assessor_count_min": "researcher_assessor_activity_count",
    "expert_assessor_count_min": "researcher_assessor_activity_count",
}

_FAMILY_VALUES = frozenset(f.value for f in Family)


def _expand_recent_doc_types(entries: Any) -> list[str]:
    """recent_doc_types 항목을 doc_type 목록으로 확장. family명이면 그 family의 doc_type 전체로."""
    out: list[str] = []
    seen: set[str] = set()
    for entry in entries or []:
        if entry in _FAMILY_VALUES:
            members = [dt for dt, fam in DOC_TYPE_TO_FAMILY.items() if fam == entry]
        elif entry in DOC_TYPES:
            members = [entry]
        else:
            members = []
        for doc_type in members:
            if doc_type not in seen:
                seen.add(doc_type)
                out.append(doc_type)
    return out


def _recency_cutoff(recent_years: Any) -> str:
    """recent_years → doc_date(datetime) 하한 문자열(RFC3339, 해당 연도 1월 1일)."""
    cutoff_year = datetime.now(UTC).year - int(recent_years)
    return f"{cutoff_year}-01-01T00:00:00Z"


class QdrantFilterCompiler:
    """hard_filters/include/exclude를 단일 Qdrant models.Filter로 컴파일한다.

    - must: 모두 만족(AND)
    - must_not: 하나라도 맞으면 제외(NOT)
    - min_should(min_count=1): 하나 이상 만족(OR) — 다중 recency 결합에 사용
    """

    def compile(
        self,
        hard_filters: dict[str, Any],
        exclude_orgs: list[str],
        include_orgs: list[str] | None = None,
    ) -> models.Filter | None:
        _ = (exclude_orgs, include_orgs)
        must: list[models.Condition] = []

        # 1. 학위 - root highest_degree
        if degree := hard_filters.get("highest_degree"):
            values = degree if isinstance(degree, list) else [degree]
            must.append(
                models.FieldCondition(
                    key="highest_degree", match=models.MatchAny(any=values)
                )
            )

        # 2. 최소 실적 건수 (flat root count >= N)
        for key, backend_key in _COUNT_MIN_TO_FIELD.items():
            min_value = hard_filters.get(key)
            if min_value is not None:
                must.append(
                    models.FieldCondition(key=backend_key, range=models.Range(gte=min_value))
                )

        # 3. journal_class 등 doc_attrs 기반 키는 의도적으로 무시한다.

        # 4. 최근성 - root doc_date(datetime) 기준. 여러 doc_type이면 OR(min_should)로 결합(0건 회귀 방지).
        recent_activity_conditions: list[models.Condition] = []
        recent_years = hard_filters.get("recent_years")
        if recent_years is not None:
            cutoff = _recency_cutoff(recent_years)
            targets = _expand_recent_doc_types(hard_filters.get("recent_doc_types"))
            if targets:
                for doc_type in targets:
                    recent_activity_conditions.append(
                        models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="doc_type", match=models.MatchValue(value=doc_type)
                                ),
                                models.FieldCondition(
                                    key="doc_date",
                                    range=models.DatetimeRange(gte=cutoff),
                                ),
                            ]
                        )
                    )
            else:
                # doc_type 미지정: 어떤 chunk든 최근 doc_date면 통과
                recent_activity_conditions.append(
                    models.FieldCondition(
                        key="doc_date", range=models.DatetimeRange(gte=cutoff)
                    )
                )

        # OR 가드 보존: 2개 이상이면 min_should(min_count=1)로 묶어 단일 must 요소로.
        if len(recent_activity_conditions) >= 2:
            must.append(
                models.Filter(
                    min_should=models.MinShould(
                        conditions=recent_activity_conditions, min_count=1
                    )
                )
            )
        elif len(recent_activity_conditions) == 1:
            must.append(recent_activity_conditions[0])

        if not must:
            return None
        return models.Filter(must=must)
