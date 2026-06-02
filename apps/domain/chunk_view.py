"""flat chunk payload에서 표시/grounding용 값을 파생하는 순수 헬퍼.

doc_attrs는 doc_type별로 키가 다르고 assessor_activity/specialty는 키가 미상(passthrough)이므로,
title/date 파생은 best-effort이고 1차 표시는 chunk_text를 사용한다. 'NONE'/빈값은 결측으로 본다.
"""
from __future__ import annotations

from typing import Any

#: 빈값 sentinel(실제 데이터의 'NONE' 문자열 포함).
_NONE_SENTINELS = {"", "none", "null", "nan", "n/a", "-", "없음"}


def clean_value(value: Any) -> Any | None:
    """문자열 'NONE'/빈값을 None으로 정규화. 비문자열은 그대로."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in _NONE_SENTINELS:
            return None
        return stripped
    return value


def normalize_doc_date(value: Any) -> str | None:
    """doc_date 문자열 정규화('NONE'/빈값 → None)."""
    cleaned = clean_value(value)
    return cleaned if isinstance(cleaned, str) else None


def parse_year(value: Any) -> int | None:
    """날짜 문자열 앞 4자리에서 연도 추출. 실패 시 None."""
    text = normalize_doc_date(value)
    if not text:
        return None
    head = text[:4]
    return int(head) if head.isdigit() else None


#: doc_type별 title 후보 키(우선순위). assessor_activity/specialty는 미상 키 best-effort.
_TITLE_KEYS: dict[str, tuple[str, ...]] = {
    "paper": ("main_language_title", "sub_language_title", "journal_name"),
    "patent": ("intellectual_property_title",),
    "project": ("project_title_korean", "project_title_english"),
    "assessor_activity": (
        "evaluation_committee_name",
        "assessment_content",
        "appointing_organization",
        "evaluation_agency_name",
    ),
    "specialty": (
        "specific_specialty_name",
        "tech_classification",
        "specialty_names",
    ),
}

#: doc_type별 대표 날짜 후보 키(root doc_date 없을 때 fallback).
_DATE_KEYS: dict[str, tuple[str, ...]] = {
    "paper": ("publication_year_month",),
    "patent": ("application_date", "registration_date"),
    "project": (),  # project_period는 아래에서 별도 파싱
    "assessor_activity": ("appointment_date", "appointment_period_start"),
    "specialty": (),
}


def _coerce_text(value: Any) -> str | None:
    if isinstance(value, list):
        joined = ", ".join(str(x).strip() for x in value if clean_value(x))
        return joined or None
    cleaned = clean_value(value)
    return str(cleaned) if cleaned is not None else None


def derive_title(doc_type: str, doc_attrs: dict[str, Any], fallback: str | None = None) -> str | None:
    """doc_attrs에서 표시용 제목 파생. 못 찾으면 fallback(보통 chunk_text 첫 줄)."""
    attrs = doc_attrs or {}
    for key in _TITLE_KEYS.get(doc_type, ()):
        title = _coerce_text(attrs.get(key))
        if title:
            return title
    if fallback:
        cleaned = clean_value(fallback)
        if isinstance(cleaned, str):
            first_line = cleaned.splitlines()[0].strip() if cleaned else ""
            return first_line or None
    return None


def derive_date(doc_type: str, doc_date: Any, doc_attrs: dict[str, Any]) -> str | None:
    """대표 날짜 파생: root doc_date 우선, 없으면 doc_type별 fallback."""
    root = normalize_doc_date(doc_date)
    if root:
        return root
    attrs = doc_attrs or {}
    for key in _DATE_KEYS.get(doc_type, ()):
        candidate = normalize_doc_date(attrs.get(key))
        if candidate:
            return candidate
    period = clean_value(attrs.get("project_period"))
    if isinstance(period, str) and "~" in period:
        start = period.split("~", 1)[0].strip()
        if start:
            return start
    return None


def snippet(chunk_text: str | None, limit: int = 280) -> str:
    """chunk_text를 표시용으로 trim."""
    text = " ".join((chunk_text or "").split())
    return text[:limit]
