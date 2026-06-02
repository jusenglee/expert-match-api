"""WO-A: 2층(도메인 통합 정규화) 유틸 — event_date/event_year, tags.

doc_type별 규칙은 표(table) 구동. DATA_MODEL.md §3.2 / §6-4 / §6-5.
"""
from __future__ import annotations

from apps.search.doc_types import DocType

# doc_type → event_date 산출에 쓸 domain_attrs 소스 필드(우선순위 순).
# 여기에 없는 doc_type은 dateless → (event_date, event_year) = (None, None).
_EVENT_SOURCE: dict[str, tuple[str, ...]] = {
    DocType.PUBLICATION: ("publication_year_month",),
    DocType.INTELLECTUAL_PROPERTY: ("registration_date", "application_date"),
    DocType.RESEARCH_PROJECT: ("research_period_start",),
    DocType.RESEARCHER_ASSESSOR: ("appointment_date", "appointment_period_start"),
    DocType.EXPERT_ASSESSOR: ("valid_period_start",),
}

#: event_date/event_year가 항상 null인 doc_type 6종 (DATA_MODEL §3.3, 샘플 7.6~7.11).
DATELESS_DOC_TYPES: frozenset[str] = frozenset(
    dt for dt in DocType if dt not in _EVENT_SOURCE
)


def normalize_tags(values: list | None) -> list[str]:
    """소문자 + trim + 중복 제거(원순서 유지). DATA_MODEL §6-5."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = (value if isinstance(value, str) else str(value)).strip().lower()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _coerce_event_date(raw: object) -> str | None:
    """원천 일자 문자열을 ISO `YYYY-MM-DD`로 정규화. None/공백 → None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if len(text) == 4 and text.isdigit():        # YYYY
        return f"{text}-01-01"
    if len(text) == 7:                            # YYYY-MM
        return f"{text}-01"
    return text[:10]                              # YYYY-MM-DD(이하 자름)


def normalize_event(doc_type: str, domain_attrs: dict | None) -> tuple[str | None, int | None]:
    """(event_date, event_year) 동시 산출 → §6-4 불일치 원천 차단. dateless 6종은 (None, None)."""
    fields = _EVENT_SOURCE.get(doc_type)
    if fields is None:
        return None, None
    attrs = domain_attrs or {}
    raw = None
    for field_name in fields:
        candidate = attrs.get(field_name)
        if candidate:
            raw = candidate
            break
    iso = _coerce_event_date(raw)
    if iso is None:
        return None, None
    try:
        year = int(iso[:4])
    except ValueError:
        return None, None
    return iso, year
