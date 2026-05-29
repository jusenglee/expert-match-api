"""WO-A: 11 doc_type별 source → chunk payload 변환.

각 doc_type은 (1) chunk_text 직렬화, (2) chunk_id/doc_id 부여(WO-0 코덱), (3) event 정규화,
(4) tags 생성, (5) researcher_meta/researcher_name 비정규화 주입을 거쳐 ChunkPayload가 된다.

HARD 제약(DATA_MODEL §2 line47 / §6-7): chunk_text에 요청 어투(role/action 불용어)를 넣지 않는다.
평가 활동 신호는 researcher_assessor/expert_assessor doc_type의 도메인 텍스트로만 표현한다.
(v1.x seed_data의 "심사평가위원 활동"/"심사참여건수" 텍스트를 복사하지 않는다.)

테이블 구동 설계: doc_type별 직렬화/태그 규칙을 표로 두고 build_chunk가 디스패치한다.
"""
from __future__ import annotations

from apps.domain.models import ChunkPayload, ResearcherMeta
from apps.ingest.normalizers import normalize_event, normalize_tags
from apps.search.doc_types import DocType, build_chunk_id, build_doc_id

__all__ = ["build_chunk", "CONVERTERS", "serialize_chunk_text", "tags_for"]


def _join(values: list | None) -> str:
    return ", ".join(str(v) for v in (values or []) if v not in (None, ""))


def _g(attrs: dict, key: str, default: object = "") -> object:
    value = attrs.get(key)
    return default if value is None else value


# --------------------------------------------------------------------------
# doc_type별 chunk_text 직렬화 (샘플 payload 7.1~7.11의 chunk_text 형식과 1:1)
#   시그니처: (researcher_name, researcher_meta, domain_attrs) -> str
# --------------------------------------------------------------------------
def _ser_publication(name, meta, a) -> str:
    return (
        f"논문명: {_g(a, 'title_primary')}\n"
        f"키워드: {_join(a.get('keywords'))}\n"
        f"학술지: {_g(a, 'journal_name')}\n"
        f"초록: {_g(a, 'abstract')}"
    )


def _ser_intellectual_property(name, meta, a) -> str:
    return (
        f"지식재산권명: {_g(a, 'ip_title')}\n"
        f"구분: {_g(a, 'ip_type')} {_g(a, 'application_registration_type')}\n"
        f"출원국가: {_g(a, 'application_country')}\n"
        f"요약: {_g(a, 'ip_summary')}"
    )


def _ser_research_project(name, meta, a) -> str:
    return (
        f"국문과제명: {_g(a, 'project_title_korean')}\n"
        f"수행기관: {_g(a, 'performing_organization')}\n"
        f"전문기관: {_g(a, 'managing_agency')}\n"
        f"국문요약: {_g(a, 'research_summary_korean')}"
    )


def _ser_researcher_assessor(name, meta, a) -> str:
    return (
        f"임명기관: {_g(a, 'appointing_organization')}\n"
        f"평가위원회명: {_g(a, 'evaluation_committee_name')}\n"
        f"임명기관구분: {_g(a, 'appointing_organization_type')}\n"
        f"활동기간구분: {_g(a, 'appointment_period_type')}"
    )


def _ser_expert_assessor(name, meta, a) -> str:
    return (
        f"평가전문기관: {_g(a, 'evaluation_agency_name')}\n"
        f"활동평가구분: {_g(a, 'assessment_class')}\n"
        f"활동평가내용: {_g(a, 'assessment_content')}\n"
        f"유형: {_g(a, 'assessment_type_class')}"
    )


def _ser_researcher_tech(name, meta, a) -> str:
    return (
        f"기술분류: {_g(a, 'tech_classification')}\n"
        f"키워드: {_join(a.get('keywords'))}\n"
        f"분류체계: {_g(a, 'tech_classification_system')}"
    )


def _ser_expert_tech(name, meta, a) -> str:
    return (
        f"기술분류: {_g(a, 'tech_classification')}\n"
        f"분류체계: {_g(a, 'tech_classification_system')}\n"
        f"기술순위: {_g(a, 'tech_rank')}"
    )


def _ser_researcher_core(name, meta, a) -> str:
    return f"핵심 전문분야: {_join(a.get('specialty_names'))}"


def _ser_researcher_major(name, meta, a) -> str:
    return f"전공 전문분야: {_join(a.get('specialty_names'))}"


def _ser_expert_specific(name, meta, a) -> str:
    return (
        f"특정전문분야: {_g(a, 'specific_specialty_name')}\n"
        f"경력: {_g(a, 'career_description')}"
    )


def _ser_profile(name, meta: ResearcherMeta, a) -> str:
    return (
        f"이름: {name}\n"
        f"소속: {meta.affiliated_organization or ''}\n"
        f"직위: {_g(a, 'position_title')}\n"
        f"학위: {_g(a, 'highest_degree')}\n"
        f"전공: {_g(a, 'major_field')}\n"
        f"논문 {_g(a, 'publication_count', 0)}편 (SCIE {_g(a, 'scie_publication_count', 0)}편), "
        f"지식재산권 {_g(a, 'intellectual_property_count', 0)}건, "
        f"과제 {_g(a, 'research_project_count', 0)}건"
    )


_SERIALIZERS = {
    DocType.PUBLICATION: _ser_publication,
    DocType.INTELLECTUAL_PROPERTY: _ser_intellectual_property,
    DocType.RESEARCH_PROJECT: _ser_research_project,
    DocType.RESEARCHER_ASSESSOR: _ser_researcher_assessor,
    DocType.EXPERT_ASSESSOR: _ser_expert_assessor,
    DocType.RESEARCHER_TECH: _ser_researcher_tech,
    DocType.EXPERT_TECH: _ser_expert_tech,
    DocType.RESEARCHER_CORE: _ser_researcher_core,
    DocType.RESEARCHER_MAJOR: _ser_researcher_major,
    DocType.EXPERT_SPECIFIC: _ser_expert_specific,
    DocType.PROFILE: _ser_profile,
}


# --------------------------------------------------------------------------
# doc_type별 tags 소스 (소문자화는 normalize_tags가 수행)
# --------------------------------------------------------------------------
def _tags_keywords(a) -> list:
    return list(a.get("keywords") or [])


def _tags_specialty(a) -> list:
    return list(a.get("specialty_names") or [])


def _tags_specific(a) -> list:
    name = a.get("specific_specialty_name")
    return [name] if name else []


_TAGS_SOURCE = {
    DocType.PUBLICATION: _tags_keywords,
    DocType.RESEARCHER_TECH: _tags_keywords,
    DocType.RESEARCHER_CORE: _tags_specialty,
    DocType.RESEARCHER_MAJOR: _tags_specialty,
    DocType.EXPERT_SPECIFIC: _tags_specific,
}  # 그 외 doc_type → 빈 배열


def serialize_chunk_text(doc_type: str, *, researcher_name: str, researcher_meta: ResearcherMeta, domain_attrs: dict) -> str:
    serializer = _SERIALIZERS.get(doc_type)
    if serializer is None:
        raise ValueError(f"unknown doc_type: {doc_type!r}")
    return serializer(researcher_name, researcher_meta, domain_attrs or {})


def tags_for(doc_type: str, domain_attrs: dict) -> list[str]:
    source = _TAGS_SOURCE.get(doc_type)
    raw = source(domain_attrs or {}) if source else []
    return normalize_tags(raw)


def build_chunk(
    doc_type: str,
    *,
    researcher_id: str,
    researcher_name: str,
    doc_seq: int | str,
    domain_attrs: dict,
    researcher_meta: ResearcherMeta,
    chunk_index: int = 0,
) -> ChunkPayload:
    """단일 소스 레코드 → ChunkPayload. chunk_id/doc_id는 WO-0 코덱으로 생성."""
    attrs = dict(domain_attrs or {})
    chunk_text = serialize_chunk_text(
        doc_type,
        researcher_name=researcher_name,
        researcher_meta=researcher_meta,
        domain_attrs=attrs,
    )
    event_date, event_year = normalize_event(doc_type, attrs)
    return ChunkPayload(
        researcher_id=researcher_id,
        researcher_name=researcher_name,
        doc_type=str(doc_type),
        doc_id=build_doc_id(doc_type, researcher_id, doc_seq),
        chunk_id=build_chunk_id(doc_type, researcher_id, doc_seq, chunk_index),
        chunk_text=chunk_text,
        chunk_text_len=len(chunk_text),
        researcher_meta=researcher_meta,
        event_date=event_date,
        event_year=event_year,
        tags=tags_for(doc_type, attrs),
        domain_attrs=attrs,
    )


#: doc_type → 변환 호출부(부분 적용). WO-C/WO-A 사용처에서 doc_type별로 호출 가능.
CONVERTERS = {
    dt: (lambda _dt: (lambda **kw: build_chunk(_dt, **kw)))(dt) for dt in DocType
}
