"""사용자 질의 분석 결과(hard_filters)를 Qdrant 필터(models.Filter)로 변환하는 모듈.

v2.0(WO-C C4): chunk 모델 기준. v1.x의 nested(`publications[]` 등) / 도메인별 recency
(`art/pat/pjt_recent_years`) / `*_cnt_min` / `basic_info.*`를 폐기하고,
DATA_CONTRACT §1.1 허용 키(`highest_degree` / `recent_years`+`recent_doc_types` /
`*_count_min`(6종) / `journal_class`)를 `researcher_meta.*` · `event_year` ·
`domain_attrs.*` chunk payload 경로로 매핑한다.

HARD 제약(보존 필수): 다중 doc_type recency는 AND가 아니라 **OR(min_should, min_count=1)**.
AND로 묶으면 0건 회귀(v1.x postmortem 교훈, DATA_MODEL §3.2).
교차-chunk org 배제(performing/managing/appointing/evaluation_agency)는 retriever 앱단
post-filter(C3) 소관이며, 여기서는 `researcher_meta.affiliated_organization`만 다룬다.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from qdrant_client import models

from apps.search.doc_types import DOC_TYPE_TO_FAMILY, DOC_TYPES, Family
from apps.search.text_utils import normalize_org_name

# 최소 건수 hard_filters 키 → researcher_meta 백엔드 필드 (DATA_CONTRACT §1.1)
_COUNT_MIN_TO_META: dict[str, str] = {
    "publication_count_min": "researcher_meta.publication_count",
    "scie_publication_count_min": "researcher_meta.scie_publication_count",
    "intellectual_property_count_min": "researcher_meta.intellectual_property_count",
    "research_project_count_min": "researcher_meta.research_project_count",
    "researcher_assessor_count_min": "researcher_meta.researcher_assessor_count",
    "expert_assessor_count_min": "researcher_meta.expert_assessor_count",
}

_AFFILIATION_KEY = "researcher_meta.affiliated_organization"
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
        must: list[models.Condition] = []
        must_not: list[models.Condition] = []

        # 0. 소속 기관 포함(include) — researcher_meta.affiliated_organization
        for org in include_orgs or []:
            normalized = normalize_org_name(org)
            if normalized:
                must.append(
                    models.FieldCondition(
                        key=_AFFILIATION_KEY, match=models.MatchValue(value=normalized)
                    )
                )

        # 1. 학위
        if degree := hard_filters.get("highest_degree"):
            values = degree if isinstance(degree, list) else [degree]
            must.append(
                models.FieldCondition(
                    key="researcher_meta.highest_degree", match=models.MatchAny(any=values)
                )
            )

        # 2. 최소 실적 건수 (researcher_meta.* >= N)
        for key, backend_key in _COUNT_MIN_TO_META.items():
            min_value = hard_filters.get(key)
            if min_value is not None:
                must.append(
                    models.FieldCondition(key=backend_key, range=models.Range(gte=min_value))
                )

        # 3. 등재구분 (publication의 domain_attrs.journal_class)
        if journal_class := hard_filters.get("journal_class"):
            values = journal_class if isinstance(journal_class, list) else [journal_class]
            must.append(
                models.FieldCondition(
                    key="domain_attrs.journal_class", match=models.MatchAny(any=values)
                )
            )

        # 4. 최근성 — 단일 event_year 기준. 여러 doc_type이면 OR(min_should)로 결합(★0건 회귀 방지).
        recent_activity_conditions: list[models.Condition] = []
        recent_years = hard_filters.get("recent_years")
        if recent_years is not None:
            cutoff_year = datetime.now(UTC).year - int(recent_years)
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
                                    key="event_year", range=models.Range(gte=cutoff_year)
                                ),
                            ]
                        )
                    )
            else:
                # doc_type 미지정: 어떤 chunk든 최근 event_year면 통과
                recent_activity_conditions.append(
                    models.FieldCondition(
                        key="event_year", range=models.Range(gte=cutoff_year)
                    )
                )

        # OR 가드 보존: 2개 이상이면 min_should(min_count=1)로 묶어 단일 must 요소로.
        # (개별 must 삽입 시 AND 교집합 → 다중 doc_type recency가 0건이 되는 v1.x 장애 재발 방지)
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

        # 5. 제외 기관 — researcher_meta.affiliated_organization
        #    (chunk의 performing/managing/appointing/evaluation_agency 교차 배제는 C3 앱단 post-filter)
        for org in exclude_orgs:
            normalized = normalize_org_name(org)
            if normalized:
                must_not.append(
                    models.FieldCondition(
                        key=_AFFILIATION_KEY, match=models.MatchValue(value=normalized)
                    )
                )

        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)
