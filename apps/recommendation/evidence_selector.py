from __future__ import annotations

import logging
from datetime import date
from typing import Protocol

from pydantic import BaseModel, Field

from apps.domain.models import (
    CandidateCard,
    IntellectualPropertyEvidence,
    PlannerOutput,
    PublicationEvidence,
    ResearchProjectEvidence,
)

logger = logging.getLogger(__name__)

RECENT_YEARS_WINDOW = 20
MAX_SELECTED_EVIDENCE_TOTAL = 4
MAX_ITEMS_PER_ASPECT = 2
SNIPPET_MAX_LENGTH = 1000


class RelevantEvidenceItem(BaseModel):
    """
    각 후보자별로 쿼리와 관련된 증거 항목(논문, 과제, 특허 등)의 기본 정보와 키워드 매칭 상태, 점수를 담는 모델입니다.
    """
    item_id: str
    type: str
    title: str
    date: str | None = None
    detail: str | None = None
    snippet: str | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    aspect_matches: list[str] = Field(default_factory=list)
    generic_matches: list[str] = Field(default_factory=list)
    direct_match: bool = False
    match_score: float = 0.0
    is_future_item: bool = False


class RelevantEvidenceBundle(BaseModel):
    """
    특정 후보자 한 명에 대해 수집된 모든 관련 증거 항목들을 묶어서 관리하는 모델입니다.
    논문, 과제, 특허 목록과 매칭된 속성 통계를 포함합니다.
    """
    expert_id: str
    papers: list[RelevantEvidenceItem] = Field(default_factory=list)
    projects: list[RelevantEvidenceItem] = Field(default_factory=list)
    patents: list[RelevantEvidenceItem] = Field(default_factory=list)
    matched_aspects: list[str] = Field(default_factory=list)
    matched_generic_terms: list[str] = Field(default_factory=list)
    direct_match_count: int = 0
    aspect_coverage: int = 0
    generic_only: bool = False
    dedup_dropped_count: int = 0
    future_selected_evidence_ids: list[str] = Field(default_factory=list)

    def all_items(self) -> list[RelevantEvidenceItem]:
        return [*self.papers, *self.projects, *self.patents]

    def by_item_id(self) -> dict[str, RelevantEvidenceItem]:
        return {item.item_id: item for item in self.all_items()}


class EvidenceSelector(Protocol):
    """
    검색 의도(PlannerOutput)에 기반하여 각 후보자에게 가장 적합한 증거 자료들을 선별하는 인터페이스입니다.
    """
    def select(
        self,
        *,
        candidates: list[CandidateCard],
        plan: PlannerOutput,
    ) -> dict[str, RelevantEvidenceBundle]: ...


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _compact_text(value: str | None) -> str:
    return _normalize_text(value).replace(" ", "")


def _normalize_title_key(value: str | None) -> str:
    return _compact_text(value)


def normalize_phrase_keywords(keywords: list[str] | None) -> list[str]:
    normalized_keywords: list[str] = []
    for keyword in keywords or []:
        normalized = _normalize_text(keyword)
        if normalized and normalized not in normalized_keywords:
            normalized_keywords.append(normalized)
    return normalized_keywords


def _build_rich_snippet(
    *,
    main_text: str | None,
    secondary_text: str | None = None,
    metadata: dict[str, str | None] | None = None,
    matched_keywords: list[str] | None = None,
) -> str | None:
    """
    문서의 본문이나 초록 등 긴 텍스트에서 매칭된 키워드가 포함된 주변 문맥을 잘라내어 요약된 스니펫(Snippet)을 생성합니다.
    LLM이 추론 시 참고하기 좋도록 글자 수 제한을 적용합니다.
    """
    parts: list[str] = []
    if metadata:
        meta_parts = [f"[{key}: {value}]" for key, value in metadata.items() if value]
        if meta_parts:
            parts.append(" ".join(meta_parts))

    text_pool = []
    if main_text:
        text_pool.append(main_text)
    if secondary_text:
        text_pool.append(secondary_text)

    full_text = " ".join(text_pool)
    normalized_text = " ".join(full_text.split())
    if not normalized_text:
        return " ".join(parts) if parts else None

    if matched_keywords and len(normalized_text) > 500:
        best_pos = -1
        for keyword in matched_keywords:
            pos = normalized_text.lower().find(keyword.lower())
            if pos != -1:
                best_pos = pos
                break
        if best_pos != -1:
            start = max(0, best_pos - 200)
            end = min(len(normalized_text), best_pos + 600)
            snippet = normalized_text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(normalized_text):
                snippet = snippet + "..."
            parts.append(snippet)
        else:
            parts.append(normalized_text[:SNIPPET_MAX_LENGTH] + "...")
    else:
        parts.append(normalized_text[:SNIPPET_MAX_LENGTH])

    return "\n".join(parts)


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value[:4])
    except (TypeError, ValueError):
        return None


def _sort_ranked_items(items: list[RelevantEvidenceItem]) -> list[RelevantEvidenceItem]:
    return sorted(
        items,
        key=lambda item: (
            not item.direct_match,
            -len(item.aspect_matches),
            -item.match_score,
            -(_parse_year(item.date) or 0),
            item.title,
            item.item_id,
        ),
    )


class KeywordEvidenceSelector:
    """
    키워드 매칭을 기반으로 후보자의 실적(논문, 과제, 특허) 중 사용자 쿼리와 가장 관련성 높은 증거를 선별하는 클래스입니다.
    """
    def __init__(self, *, reference_year: int | None = None) -> None:
        self.reference_year = reference_year or date.today().year
        self.last_trace: dict[str, object] = {}

    def select(
        self,
        *,
        candidates: list[CandidateCard],
        plan: PlannerOutput,
    ) -> dict[str, RelevantEvidenceBundle]:
        """
        주어진 후보자 목록 각각에 대하여 논문, 과제, 특허 중 쿼리와 일치하는 증거 자료를 선별하고, 
        중복을 제거한 뒤 가장 관련성이 높은 항목들만 묶어서(RelevantEvidenceBundle) 반환합니다.
        """
        # evidence_aspects: 한국어+영어 혼합 매칭 용어 (planner v0.7.0+ 생성).
        # 없으면 한국어 must_aspects → retrieval_core 순으로 폴백.
        # evidence_aspects는 실제 논문/과제/특허 텍스트에 등장하는 표현을 포함하므로
        # Korean/English 혼합 데이터베이스에서 직접 매칭이 동작한다.
        aspects = normalize_phrase_keywords(
            plan.evidence_aspects
            or plan.must_aspects
            or plan.retrieval_core
            or plan.core_keywords
        )
        generic_terms = normalize_phrase_keywords(plan.generic_terms)
        bundles: dict[str, RelevantEvidenceBundle] = {}
        candidate_diagnostics: list[dict[str, object]] = []
        empty_candidate_ids: list[str] = []

        for candidate in candidates:
            ranked_items = [
                *self._rank_publications(candidate.top_papers, aspects, generic_terms),
                *self._rank_projects(candidate.top_projects, aspects, generic_terms),
                *self._rank_patents(candidate.top_patents, aspects, generic_terms),
            ]
            deduped_items, dedup_dropped_count = self._deduplicate_items(ranked_items)
            selected_items = self._select_direct_evidence(deduped_items, aspects)
            matched_aspects = self._collect_unique_keywords(
                [match for item in selected_items for match in item.aspect_matches]
            )
            matched_generic_terms = self._collect_unique_keywords(
                [match for item in deduped_items for match in item.generic_matches]
            )
            direct_match_count = sum(1 for item in deduped_items if item.direct_match)
            bundle = RelevantEvidenceBundle(
                expert_id=candidate.expert_id,
                papers=[item for item in selected_items if item.type == "paper"],
                projects=[item for item in selected_items if item.type == "project"],
                patents=[item for item in selected_items if item.type == "patent"],
                matched_aspects=matched_aspects,
                matched_generic_terms=matched_generic_terms,
                direct_match_count=direct_match_count,
                aspect_coverage=len(matched_aspects),
                generic_only=bool(matched_generic_terms) and direct_match_count == 0,
                dedup_dropped_count=dedup_dropped_count,
                future_selected_evidence_ids=[
                    item.item_id for item in selected_items if item.is_future_item
                ],
            )
            bundles[candidate.expert_id] = bundle

            if not selected_items:
                empty_candidate_ids.append(candidate.expert_id)

            diagnostic = {
                "expert_id": candidate.expert_id,
                "selected_evidence_ids": [item.item_id for item in selected_items],
                "direct_match_count": direct_match_count,
                "aspect_coverage": bundle.aspect_coverage,
                "generic_only": bundle.generic_only,
                "matched_aspects": matched_aspects,
                "matched_generic_terms": matched_generic_terms,
                "dedup_dropped_count": dedup_dropped_count,
                "future_selected_evidence_ids": list(
                    bundle.future_selected_evidence_ids
                ),
                "papers": len(bundle.papers),
                "projects": len(bundle.projects),
                "patents": len(bundle.patents),
                "total": len(selected_items),
            }
            candidate_diagnostics.append(diagnostic)
            logger.info(
                "Evidence selection candidate summary: expert_id=%s direct_match_count=%d aspect_coverage=%d generic_only=%s selected_evidence_ids=%s future_selected_evidence_ids=%s dedup_dropped_count=%d",
                candidate.expert_id,
                direct_match_count,
                bundle.aspect_coverage,
                bundle.generic_only,
                diagnostic["selected_evidence_ids"],
                diagnostic["future_selected_evidence_ids"],
                dedup_dropped_count,
            )

        aspect_source = (
            "evidence_aspects" if plan.evidence_aspects
            else "must_aspects" if plan.must_aspects
            else "retrieval_core"
        )
        self.last_trace = {
            "mode": "direct_match_aspect_quota",
            "aspect_source": aspect_source,
            "aspects": aspects,
            "generic_terms": generic_terms,
            "candidate_evidence_counts": candidate_diagnostics,
            "empty_candidate_ids": empty_candidate_ids,
        }
        logger.info(
            "Evidence selection completed: candidates=%d empty_candidates=%d aspect_source=%s aspects=%s",
            len(candidates),
            len(empty_candidate_ids),
            aspect_source,
            aspects,
        )
        return bundles

    @staticmethod
    def _normalize_keywords(keywords: list[str]) -> list[str]:
        return normalize_phrase_keywords(keywords)

    @staticmethod
    def _collect_unique_keywords(values: list[str]) -> list[str]:
        collected: list[str] = []
        for value in values:
            normalized = _normalize_text(value)
            if normalized and normalized not in collected:
                collected.append(normalized)
        return collected

    def _rank_publications(
        self,
        publications: list[PublicationEvidence],
        aspects: list[str],
        generic_terms: list[str],
    ) -> list[RelevantEvidenceItem]:
        ranked: list[RelevantEvidenceItem] = []
        for index, item in enumerate(publications):
            base_score = 1.0 + (len(publications) - index) * 0.1
            score, aspect_matches, generic_matches = self._score_evidence(
                title=item.publication_title,
                body_parts=[
                    item.journal_name,
                    item.abstract,
                    " ".join(item.korean_keywords),
                    " ".join(item.english_keywords),
                ],
                date_value=item.publication_year_month,
                aspects=aspects,
                generic_terms=generic_terms,
            )
            ranked.append(
                RelevantEvidenceItem(
                    item_id=f"paper:{index}",
                    type="paper",
                    title=item.publication_title,
                    date=item.publication_year_month,
                    detail=item.journal_name,
                    snippet=_build_rich_snippet(
                        main_text=item.abstract,
                        secondary_text=" ".join(item.korean_keywords + item.english_keywords),
                        metadata={"journal": item.journal_name},
                        matched_keywords=aspect_matches or generic_matches,
                    ),
                    matched_keywords=self._collect_unique_keywords(
                        [*aspect_matches, *generic_matches]
                    ),
                    aspect_matches=aspect_matches,
                    generic_matches=generic_matches,
                direct_match=bool(aspect_matches),
                match_score=base_score + score,
                is_future_item=False,
            )
            )
        return _sort_ranked_items(ranked)

    def _rank_projects(
        self,
        projects: list[ResearchProjectEvidence],
        aspects: list[str],
        generic_terms: list[str],
    ) -> list[RelevantEvidenceItem]:
        ranked: list[RelevantEvidenceItem] = []
        for index, item in enumerate(projects):
            base_score = 1.0 + (len(projects) - index) * 0.1
            # project_title_english을 body_parts에 추가.
            # display_title은 한국어 제목을 우선하므로, 영어 제목이 검색에서 누락될 수 있음.
            # evidence_aspects에 영어 용어가 포함된 경우 영어 제목에서도 매칭되어야 함.
            en_title_supplement = (
                item.project_title_english
                if item.project_title_english and item.project_title_english != item.display_title
                else None
            )
            score, aspect_matches, generic_matches = self._score_evidence(
                title=item.display_title,
                body_parts=[
                    en_title_supplement,
                    item.research_objective_summary,
                    item.research_content_summary,
                    item.managing_agency,
                    item.performing_organization,
                ],
                date_value=item.project_end_date or item.project_start_date,
                aspects=aspects,
                generic_terms=generic_terms,
            )
            ranked.append(
                RelevantEvidenceItem(
                    item_id=f"project:{index}",
                    type="project",
                    title=item.display_title,
                    date=item.project_end_date or item.project_start_date,
                    detail=item.managing_agency or item.performing_organization,
                    snippet=_build_rich_snippet(
                        main_text=item.research_objective_summary,
                        secondary_text=item.research_content_summary,
                        metadata={
                            "organization": item.performing_organization
                            or item.managing_agency
                        },
                        matched_keywords=aspect_matches or generic_matches,
                    ),
                    matched_keywords=self._collect_unique_keywords(
                        [*aspect_matches, *generic_matches]
                    ),
                    aspect_matches=aspect_matches,
                    generic_matches=generic_matches,
                direct_match=bool(aspect_matches),
                match_score=base_score + score,
                is_future_item=bool(
                    (year := _parse_year(item.project_end_date or item.project_start_date))
                    and year > self.reference_year
                ),
            )
            )
        return _sort_ranked_items(ranked)

    def _rank_patents(
        self,
        patents: list[IntellectualPropertyEvidence],
        aspects: list[str],
        generic_terms: list[str],
    ) -> list[RelevantEvidenceItem]:
        ranked: list[RelevantEvidenceItem] = []
        for index, item in enumerate(patents):
            base_score = 1.0 + (len(patents) - index) * 0.1
            score, aspect_matches, generic_matches = self._score_evidence(
                title=item.intellectual_property_title,
                body_parts=[
                    item.application_registration_type,
                    item.application_country,
                ],
                date_value=item.registration_date or item.application_date,
                aspects=aspects,
                generic_terms=generic_terms,
            )
            ranked.append(
                RelevantEvidenceItem(
                    item_id=f"patent:{index}",
                    type="patent",
                    title=item.intellectual_property_title,
                    date=item.registration_date or item.application_date,
                    detail=item.application_registration_type
                    or item.application_country,
                    snippet=_build_rich_snippet(
                        main_text=item.intellectual_property_title,
                        secondary_text=f"type: {item.application_registration_type}, country: {item.application_country}",
                        matched_keywords=aspect_matches or generic_matches,
                    ),
                    matched_keywords=self._collect_unique_keywords(
                        [*aspect_matches, *generic_matches]
                    ),
                    aspect_matches=aspect_matches,
                    generic_matches=generic_matches,
                direct_match=bool(aspect_matches),
                match_score=base_score + score,
                is_future_item=False,
            )
            )
        return _sort_ranked_items(ranked)

    def _score_evidence(
        self,
        *,
        title: str | None,
        body_parts: list[str | None],
        date_value: str | None,
        aspects: list[str],
        generic_terms: list[str],
    ) -> tuple[float, list[str], list[str]]:
        """
        단일 증거 항목(논문, 특허 등)의 텍스트(제목, 본문)를 분석하여 키워드 매칭 점수와 일치한 키워드 목록을 반환합니다.
        - 제목에서 매칭될 경우 본문 매칭보다 더 높은 점수(가중치)를 부여합니다.
        - 최신 연도(최근 20년 이내) 실적일 경우 추가 점수를 줍니다.
        """
        normalized_title = _normalize_text(title)
        compact_title = _compact_text(title)
        normalized_body = _normalize_text(" ".join(part or "" for part in body_parts))
        compact_body = _compact_text(" ".join(part or "" for part in body_parts))

        score = 0.0
        aspect_matches: list[str] = []
        generic_matches: list[str] = []

        for aspect in aspects:
            compact_aspect = aspect.replace(" ", "")
            matched = False
            if normalized_title == aspect or compact_title == compact_aspect:
                score += 8.0
                matched = True
            elif aspect in normalized_title or compact_aspect in compact_title:
                score += 5.0
                matched = True

            if aspect in normalized_body or compact_aspect in compact_body:
                score += 3.0
                matched = True

            if matched and aspect not in aspect_matches:
                aspect_matches.append(aspect)

        for generic_term in generic_terms:
            compact_term = generic_term.replace(" ", "")
            if (
                generic_term in normalized_title
                or compact_term in compact_title
                or generic_term in normalized_body
                or compact_term in compact_body
            ) and generic_term not in generic_matches:
                generic_matches.append(generic_term)

        if len(aspect_matches) > 1:
            score += float(len(aspect_matches) - 1)

        year = _parse_year(date_value)
        if (
            aspect_matches
            and year is not None
            and year >= self.reference_year - (RECENT_YEARS_WINDOW - 1)
        ):
            score += 0.5

        return score, aspect_matches, generic_matches

    @staticmethod
    def _deduplicate_items(
        items: list[RelevantEvidenceItem],
    ) -> tuple[list[RelevantEvidenceItem], int]:
        deduped: dict[tuple[str, int | None], RelevantEvidenceItem] = {}
        dedup_dropped_count = 0
        for item in _sort_ranked_items(items):
            dedup_key = (_normalize_title_key(item.title), _parse_year(item.date))
            if dedup_key in deduped:
                dedup_dropped_count += 1
                continue
            deduped[dedup_key] = item
        return _sort_ranked_items(list(deduped.values())), dedup_dropped_count

    @staticmethod
    def _append_unique_item(
        selected_items: list[RelevantEvidenceItem],
        item: RelevantEvidenceItem,
    ) -> bool:
        if item.item_id in {selected.item_id for selected in selected_items}:
            return False
        selected_items.append(item)
        return True

    def _select_direct_evidence(
        self,
        items: list[RelevantEvidenceItem],
        aspects: list[str],
    ) -> list[RelevantEvidenceItem]:
        """
        중복이 제거된 증거 목록 중에서, 사용자 쿼리 키워드(aspect)와 직접 매칭(direct match)된 증거들만 최종 선택합니다.
        다양한 키워드들이 고르게 선택될 수 있도록 항목 개수(MAX_SELECTED_EVIDENCE_TOTAL)를 제한합니다.
        """
        direct_items = [item for item in items if item.direct_match]
        if not direct_items:
            return []

        selected_items: list[RelevantEvidenceItem] = []
        aspect_counts = {aspect: 0 for aspect in aspects}

        for aspect in aspects:
            for item in direct_items:
                if aspect not in item.aspect_matches or aspect_counts[aspect] >= 1:
                    continue
                if self._append_unique_item(selected_items, item):
                    for matched_aspect in item.aspect_matches:
                        if matched_aspect in aspect_counts:
                            aspect_counts[matched_aspect] += 1
                if len(selected_items) >= MAX_SELECTED_EVIDENCE_TOTAL:
                    return _sort_ranked_items(selected_items)
                if aspect_counts[aspect] >= 1:
                    break

        for item in direct_items:
            if len(selected_items) >= MAX_SELECTED_EVIDENCE_TOTAL:
                break
            if any(
                aspect_counts.get(aspect, 0) >= MAX_ITEMS_PER_ASPECT
                for aspect in item.aspect_matches
            ):
                continue
            if self._append_unique_item(selected_items, item):
                for matched_aspect in item.aspect_matches:
                    if matched_aspect in aspect_counts:
                        aspect_counts[matched_aspect] += 1

        return _sort_ranked_items(selected_items)
