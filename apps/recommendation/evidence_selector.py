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

RECENT_YEARS_WINDOW = 20
MAX_RELEVANT_PAPERS = 10
MAX_RELEVANT_PROJECTS = 10
MAX_RELEVANT_PATENTS = 10
SNIPPET_MAX_LENGTH = 1000


class RelevantEvidenceItem(BaseModel):
    item_id: str
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

    def all_items(self) -> list[RelevantEvidenceItem]:
        return [*self.papers, *self.projects, *self.patents]

    def by_item_id(self) -> dict[str, RelevantEvidenceItem]:
        return {item.item_id: item for item in self.all_items()}


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


def _build_rich_snippet(
    *,
    main_text: str | None,
    secondary_text: str | None = None,
    metadata: dict[str, str | None] | None = None,
    matched_keywords: list[str] | None = None,
) -> str | None:
    parts: list[str] = []
    
    # 메타데이터 추가 (예: 학술지명, 기관명 등)
    if metadata:
        meta_parts = [f"[{k}: {v}]" for k, v in metadata.items() if v]
        if meta_parts:
            parts.append(" ".join(meta_parts))

    # 주요 텍스트 추가
    text_pool = []
    if main_text:
        text_pool.append(main_text)
    if secondary_text:
        text_pool.append(secondary_text)
    
    full_text = " ".join(text_pool)
    normalized_text = " ".join(full_text.split())
    
    if not normalized_text:
        return " ".join(parts) if parts else None

    # 키워드 주변 문맥 추출 (간소화된 버전)
    if matched_keywords and len(normalized_text) > 500:
        # 첫 번째 매칭된 키워드 위치 찾기
        best_pos = -1
        for kw in matched_keywords:
            pos = normalized_text.lower().find(kw.lower())
            if pos != -1:
                best_pos = pos
                break
        
        if best_pos != -1:
            start = max(0, best_pos - 200)
            end = min(len(normalized_text), best_pos + 600)
            snippet = normalized_text[start:end]
            if start > 0: snippet = "..." + snippet
            if end < len(normalized_text): snippet = snippet + "..."
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
            # 개별 브랜치에서 후보군 랭킹 (내부 limit는 넉넉히 유지)
            papers = self._rank_publications(candidate.top_papers, keywords)
            projects = self._rank_projects(candidate.top_projects, keywords)
            patents = self._rank_patents(candidate.top_patents, keywords)
            
            # 모든 증거를 합쳐서 매칭 점수순으로 최상위 3개만 엄선
            # 이는 LLM(Reasoner)에게 노이즈 없는 핵심 데이터만 전달하기 위함임
            top_evidence = sorted(
                [*papers, *projects, *patents],
                key=lambda item: -item.match_score
            )[:3]

            bundle = RelevantEvidenceBundle(
                expert_id=candidate.expert_id,
                papers=[e for e in top_evidence if e.type == "paper"],
                projects=[e for e in top_evidence if e.type == "project"],
                patents=[e for e in top_evidence if e.type == "patent"],
            )
            bundles[candidate.expert_id] = bundle

            candidate_count = len(top_evidence)
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
            "mode": "keyword_lexical_top3",
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
        for index, item in enumerate(publications):
            base_score = 1.0 + (len(publications) - index) * 0.1
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
            final_score = base_score + score

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
                        metadata={"학술지": item.journal_name},
                        matched_keywords=matched_keywords,
                    ),
                    matched_keywords=matched_keywords,
                    match_score=final_score,
                )
            )
        return self._finalize_ranked_items(ranked, MAX_RELEVANT_PAPERS)

    def _rank_projects(
        self,
        projects: list[ResearchProjectEvidence],
        keywords: list[str],
    ) -> list[RelevantEvidenceItem]:
        ranked: list[RelevantEvidenceItem] = []
        for index, item in enumerate(projects):
            base_score = 1.0 + (len(projects) - index) * 0.1
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
            final_score = base_score + score

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
                        metadata={"기관": item.performing_organization or item.managing_agency},
                        matched_keywords=matched_keywords,
                    ),
                    matched_keywords=matched_keywords,
                    match_score=final_score,
                )
            )
        return self._finalize_ranked_items(ranked, MAX_RELEVANT_PROJECTS)

    def _rank_patents(
        self,
        patents: list[IntellectualPropertyEvidence],
        keywords: list[str],
    ) -> list[RelevantEvidenceItem]:
        ranked: list[RelevantEvidenceItem] = []
        for index, item in enumerate(patents):
            base_score = 1.0 + (len(patents) - index) * 0.1
            score, matched_keywords = self._score_evidence(
                title=item.intellectual_property_title,
                body_parts=[
                    item.application_registration_type,
                    item.application_country,
                ],
                date_value=item.registration_date or item.application_date,
                keywords=keywords,
            )
            final_score = base_score + score

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
                        secondary_text=f"유형: {item.application_registration_type}, 국가: {item.application_country}",
                        matched_keywords=matched_keywords,
                    ),
                    matched_keywords=matched_keywords,
                    match_score=final_score,
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
