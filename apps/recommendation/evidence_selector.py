from __future__ import annotations

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

RECENT_YEARS_WINDOW = 5
MAX_RELEVANT_PAPERS = 2
MAX_RELEVANT_PROJECTS = 2
MAX_RELEVANT_PATENTS = 1
SNIPPET_MAX_LENGTH = 180


class RelevantEvidenceItem(BaseModel):
    type: str
    title: str
    date: str | None = None
    detail: str | None = None
    snippet: str | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    match_score: float = 0.0


class RelevantEvidenceBundle(BaseModel):
    expert_id: str
    papers: list[RelevantEvidenceItem] = Field(default_factory=list)
    projects: list[RelevantEvidenceItem] = Field(default_factory=list)
    patents: list[RelevantEvidenceItem] = Field(default_factory=list)


class EvidenceSelector(Protocol):
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


def _build_snippet(*values: str | None) -> str | None:
    for value in values:
        normalized = " ".join((value or "").split())
        if not normalized:
            continue
        if len(normalized) <= SNIPPET_MAX_LENGTH:
            return normalized
        return normalized[: SNIPPET_MAX_LENGTH - 3].rstrip() + "..."
    return None


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value[:4])
    except (TypeError, ValueError):
        return None


class KeywordEvidenceSelector:
    def __init__(self, *, reference_year: int | None = None) -> None:
        self.reference_year = reference_year or date.today().year
        self.last_trace: dict[str, object] = {}

    def select(
        self,
        *,
        candidates: list[CandidateCard],
        plan: PlannerOutput,
    ) -> dict[str, RelevantEvidenceBundle]:
        keywords = self._normalize_keywords(plan.core_keywords)
        bundles: dict[str, RelevantEvidenceBundle] = {}
        candidate_evidence_counts: list[dict[str, object]] = []
        empty_candidate_ids: list[str] = []

        for candidate in candidates:
            bundle = RelevantEvidenceBundle(
                expert_id=candidate.expert_id,
                papers=self._rank_publications(candidate.top_papers, keywords),
                projects=self._rank_projects(candidate.top_projects, keywords),
                patents=self._rank_patents(candidate.top_patents, keywords),
            )
            bundles[candidate.expert_id] = bundle

            candidate_count = (
                len(bundle.papers) + len(bundle.projects) + len(bundle.patents)
            )
            if candidate_count == 0:
                empty_candidate_ids.append(candidate.expert_id)
            candidate_evidence_counts.append(
                {
                    "expert_id": candidate.expert_id,
                    "papers": len(bundle.papers),
                    "projects": len(bundle.projects),
                    "patents": len(bundle.patents),
                    "total": candidate_count,
                }
            )

        self.last_trace = {
            "mode": "keyword_lexical",
            "core_keywords": keywords,
            "candidate_evidence_counts": candidate_evidence_counts,
            "empty_candidate_ids": empty_candidate_ids,
        }
        return bundles

    @staticmethod
    def _normalize_keywords(keywords: list[str]) -> list[str]:
        normalized_keywords: list[str] = []
        for keyword in keywords:
            normalized = _normalize_text(keyword)
            if normalized and normalized not in normalized_keywords:
                normalized_keywords.append(normalized)
        return normalized_keywords

    def _rank_publications(
        self,
        publications: list[PublicationEvidence],
        keywords: list[str],
    ) -> list[RelevantEvidenceItem]:
        ranked: list[RelevantEvidenceItem] = []
        for item in publications:
            score, matched_keywords = self._score_evidence(
                title=item.publication_title,
                body_parts=[
                    item.journal_name,
                    item.abstract,
                    " ".join(item.korean_keywords),
                    " ".join(item.english_keywords),
                ],
                date_value=item.publication_year_month,
                keywords=keywords,
            )
            if score <= 0:
                continue
            ranked.append(
                RelevantEvidenceItem(
                    type="paper",
                    title=item.publication_title,
                    date=item.publication_year_month,
                    detail=item.journal_name,
                    snippet=_build_snippet(item.abstract),
                    matched_keywords=matched_keywords,
                    match_score=score,
                )
            )
        return self._finalize_ranked_items(ranked, MAX_RELEVANT_PAPERS)

    def _rank_projects(
        self,
        projects: list[ResearchProjectEvidence],
        keywords: list[str],
    ) -> list[RelevantEvidenceItem]:
        ranked: list[RelevantEvidenceItem] = []
        for item in projects:
            score, matched_keywords = self._score_evidence(
                title=item.display_title,
                body_parts=[
                    item.research_objective_summary,
                    item.research_content_summary,
                    item.managing_agency,
                    item.performing_organization,
                ],
                date_value=item.project_end_date or item.project_start_date,
                keywords=keywords,
            )
            if score <= 0:
                continue
            ranked.append(
                RelevantEvidenceItem(
                    type="project",
                    title=item.display_title,
                    date=item.project_end_date or item.project_start_date,
                    detail=item.managing_agency or item.performing_organization,
                    snippet=_build_snippet(
                        item.research_objective_summary,
                        item.research_content_summary,
                    ),
                    matched_keywords=matched_keywords,
                    match_score=score,
                )
            )
        return self._finalize_ranked_items(ranked, MAX_RELEVANT_PROJECTS)

    def _rank_patents(
        self,
        patents: list[IntellectualPropertyEvidence],
        keywords: list[str],
    ) -> list[RelevantEvidenceItem]:
        ranked: list[RelevantEvidenceItem] = []
        for item in patents:
            score, matched_keywords = self._score_evidence(
                title=item.intellectual_property_title,
                body_parts=[
                    item.application_registration_type,
                    item.application_country,
                ],
                date_value=item.registration_date or item.application_date,
                keywords=keywords,
            )
            if score <= 0:
                continue
            ranked.append(
                RelevantEvidenceItem(
                    type="patent",
                    title=item.intellectual_property_title,
                    date=item.registration_date or item.application_date,
                    detail=item.application_registration_type
                    or item.application_country,
                    matched_keywords=matched_keywords,
                    match_score=score,
                )
            )
        return self._finalize_ranked_items(ranked, MAX_RELEVANT_PATENTS)

    def _score_evidence(
        self,
        *,
        title: str | None,
        body_parts: list[str | None],
        date_value: str | None,
        keywords: list[str],
    ) -> tuple[float, list[str]]:
        if not keywords:
            return 0.0, []

        normalized_title = _normalize_text(title)
        compact_title = _compact_text(title)
        normalized_body = _normalize_text(" ".join(part or "" for part in body_parts))
        compact_body = _compact_text(" ".join(part or "" for part in body_parts))

        score = 0.0
        matched_keywords: list[str] = []

        for keyword in keywords:
            compact_keyword = keyword.replace(" ", "")
            matched = False
            if normalized_title == keyword or compact_title == compact_keyword:
                score += 8.0
                matched = True
            elif keyword in normalized_title or compact_keyword in compact_title:
                score += 5.0
                matched = True

            if keyword in normalized_body or compact_keyword in compact_body:
                score += 3.0
                matched = True

            if matched and keyword not in matched_keywords:
                matched_keywords.append(keyword)

        if len(matched_keywords) > 1:
            score += float(len(matched_keywords) - 1)

        year = _parse_year(date_value)
        if (
            matched_keywords
            and year is not None
            and year >= self.reference_year - (RECENT_YEARS_WINDOW - 1)
        ):
            score += 0.5

        return score, matched_keywords

    @staticmethod
    def _finalize_ranked_items(
        items: list[RelevantEvidenceItem],
        limit: int,
    ) -> list[RelevantEvidenceItem]:
        ranked = sorted(
            items,
            key=lambda item: (
                -item.match_score,
                -(_parse_year(item.date) or 0),
                item.title,
            ),
        )
        return ranked[:limit]
