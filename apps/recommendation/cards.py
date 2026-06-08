"""검색 집계 결과(ResearcherCandidate)로부터 후보 카드(CandidateCard)를 만든다.

flat chunk 모델(v2.1): 1 chunk = 1 doc/evidence. 정체성/누적 실적은 chunk root에서 가져오고,
evidence는 doc_type별 ChunkEvidence 묶음으로 구성한다(assessor_activity/specialty 포함).
"""

from __future__ import annotations

from typing import Any

from apps.domain.chunk_view import derive_date, derive_title, parse_year, snippet
from apps.domain.models import (
    CandidateCard,
    ChunkEvidence,
    PlannerOutput,
    ResearcherCandidate,
)

# 후보 카드 count 표시 키 ← flat root count 필드.
_COUNT_DISPLAY: dict[str, str] = {
    "article_cnt": "publication_count",
    "scie_cnt": "scie_publication_count",
    "patent_cnt": "intellectual_property_count",
    "project_cnt": "research_project_count",
    "assessor_cnt": "researcher_assessor_activity_count",
}

# data_gap 검사: count 필드 → 사람이 읽는 라벨.
_GAP_LABELS: dict[str, str] = {
    "publication_count": "Publication evidence is missing.",
    "intellectual_property_count": "Patent evidence is missing.",
    "research_project_count": "Project evidence is missing.",
}


def _evidence_sort_key(ev: ChunkEvidence) -> tuple[int, float]:
    return (parse_year(ev.date) or 0, ev.score)


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


class CandidateCardBuilder:
    def build_small_cards(
        self, hits: list[ResearcherCandidate], plan: PlannerOutput
    ) -> list[CandidateCard]:
        if not hits:
            return []
        cards = [self._build_card(hit, plan) for hit in hits]
        max_score = max((hit.group_score for hit in hits), default=0.0)
        for index, card in enumerate(cards):
            raw_score = hits[index].group_score
            normalized = (raw_score / max_score * 100) if max_score else 0.0
            card.rank_score = round(float(normalized), 1)
            card.shortlist_score = card.rank_score
        return cards

    def shortlist(self, cards: list[CandidateCard], limit: int) -> list[CandidateCard]:
        return cards[:limit]

    def _build_evidence_by_type(
        self, candidate: ResearcherCandidate
    ) -> dict[str, list[ChunkEvidence]]:
        buckets: dict[str, list[ChunkEvidence]] = {}
        for hit in candidate.chunks:
            payload = hit.payload
            doc_type = payload.doc_type
            evidence = ChunkEvidence(
                chunk_id=payload.chunk_id,
                doc_type=doc_type,
                title=derive_title(doc_type, payload.doc_attrs, fallback=payload.chunk_text),
                date=derive_date(doc_type, payload.doc_date, payload.doc_attrs),
                snippet=snippet(payload.chunk_text),
                doc_attrs=payload.doc_attrs,
                score=hit.score,
            )
            buckets.setdefault(doc_type, []).append(evidence)
        for items in buckets.values():
            items.sort(key=_evidence_sort_key, reverse=True)
        return buckets

    def _build_top_chunks(self, candidate: ResearcherCandidate) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for hit in candidate.chunks[:10]:
            payload = hit.payload
            chunks.append(
                {
                    "chunk_id": payload.chunk_id,
                    "doc_type": payload.doc_type,
                    "title": derive_title(
                        payload.doc_type,
                        payload.doc_attrs,
                        fallback=payload.chunk_text,
                    ),
                    "concepts": list(hit.concepts),
                    "sources": list(hit.sources),
                    "score": round(float(hit.score), 6),
                }
            )
        return chunks

    def _build_card(self, candidate: ResearcherCandidate, plan: PlannerOutput) -> CandidateCard:
        counts_root = candidate.counts
        counts = {
            display: int(counts_root.get(field, 0) or 0)
            for display, field in _COUNT_DISPLAY.items()
        }

        # matched_filter_summary: canonical hard_filter 키(filters.py와 정합) 기반.
        hard_filters = plan.hard_filters
        matched_filter_summary: list[str] = []
        if hard_filters.get("highest_degree"):
            matched_filter_summary.append(
                f"Degree filter matched: {candidate.highest_degree or 'unknown'}"
            )
        if hard_filters.get("research_project_count_min") is not None:
            matched_filter_summary.append(f"Project count: {counts['project_cnt']}")

        # data_gaps / risks: root count 기반(개별 chunk 배열이 아니라 누적 실적).
        data_gaps: list[str] = [
            label for field, label in _GAP_LABELS.items() if not counts_root.get(field)
        ]
        risks: list[str] = []
        if len(data_gaps) >= 2:
            risks.append("Evidence coverage is limited.")

        return CandidateCard(
            expert_id=candidate.researcher_id,
            name=candidate.researcher_name,
            organization=candidate.affiliated_organization,
            degree=candidate.highest_degree,
            counts=counts,
            evidence_by_type=self._build_evidence_by_type(candidate),
            matched_filter_summary=matched_filter_summary,
            risks=risks,
            data_gaps=data_gaps,
            raw_score=round(float(candidate.group_score), 6),
            matched_concepts=list(candidate.matched_concepts),
            missing_concepts=_unique_sorted(
                [
                    *candidate.missing_concepts,
                    *[
                        concept
                        for concept in plan.required_concepts
                        if concept not in candidate.matched_concepts
                    ],
                ]
            ),
            coverage_type=candidate.coverage_type,
            score_breakdown=dict(candidate.score_breakdown),
            top_chunks=self._build_top_chunks(candidate),
        )
