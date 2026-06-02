"""WO-A: researcher_meta(8필드) 집계 + 비정규화 주입 (DATA_MODEL §3.1 / §6-2).

정책: profile doc의 count를 1차 truth로 사용하고, 부재 시 doc_type별 레코드 수를 fallback으로 센다.
profile count와 실제 레코드 수가 어긋나면 meta_discrepancies로 노출(검증/경고). 어느 값을 truth로
둘지의 최종 결정은 데이터 제공처 확인 후 WO-0 계약에 반영(open question).
"""
from __future__ import annotations

from apps.domain.models import ResearcherMeta
from apps.search.doc_types import DocType

_COUNT_FIELDS: tuple[str, ...] = (
    "publication_count",
    "scie_publication_count",
    "intellectual_property_count",
    "research_project_count",
    "researcher_assessor_count",
    "expert_assessor_count",
)

# doc_type → 레코드 수가 누적되는 count 필드(레코드 fallback 집계용).
# scie_publication_count는 별도 doc_type이 없어 fallback 0(profile truth에 의존).
_DOC_TYPE_COUNT_FIELD = {
    DocType.PUBLICATION: "publication_count",
    DocType.INTELLECTUAL_PROPERTY: "intellectual_property_count",
    DocType.RESEARCH_PROJECT: "research_project_count",
    DocType.RESEARCHER_ASSESSOR: "researcher_assessor_count",
    DocType.EXPERT_ASSESSOR: "expert_assessor_count",
}


def count_records_by_doc_type(doc_types) -> dict[str, int]:
    """doc_type 시퀀스 → {count_field: n}. 한 연구자의 레코드 doc_type 목록을 넣는다."""
    counts = {field: 0 for field in _COUNT_FIELDS}
    for doc_type in doc_types:
        field = _DOC_TYPE_COUNT_FIELD.get(doc_type)
        if field is not None:
            counts[field] += 1
    return counts


def build_researcher_meta(
    *,
    profile_attrs: dict | None = None,
    affiliated_organization: str | None = None,
    record_counts: dict[str, int] | None = None,
) -> ResearcherMeta:
    """연구자 1명의 researcher_meta 산출. 한 연구자의 모든 chunk에 동일 주입한다(§6-2).

    affiliated_organization은 profile domain_attrs에 없으므로(샘플 7.11) 외부 인자로 받는다.
    highest_degree는 profile domain_attrs에서.
    """
    profile = profile_attrs or {}
    records = record_counts or {}

    def pick(field: str) -> int:
        value = profile.get(field)
        if value is not None:
            return value
        return records.get(field, 0)

    return ResearcherMeta(
        affiliated_organization=affiliated_organization,
        highest_degree=profile.get("highest_degree"),
        publication_count=pick("publication_count"),
        scie_publication_count=pick("scie_publication_count"),
        intellectual_property_count=pick("intellectual_property_count"),
        research_project_count=pick("research_project_count"),
        researcher_assessor_count=pick("researcher_assessor_count"),
        expert_assessor_count=pick("expert_assessor_count"),
    )


def meta_discrepancies(
    profile_attrs: dict | None, record_counts: dict[str, int] | None
) -> dict[str, dict[str, int]]:
    """profile count vs 레코드 수 불일치 항목(검증/경고용). 빈 dict면 일치."""
    profile = profile_attrs or {}
    records = record_counts or {}
    out: dict[str, dict[str, int]] = {}
    for field in _COUNT_FIELDS:
        profile_value = profile.get(field)
        if profile_value is None or field not in records:
            continue
        if profile_value != records[field]:
            out[field] = {"profile": profile_value, "records": records[field]}
    return out
