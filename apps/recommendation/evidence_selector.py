from __future__ import annotations

import math
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
    # 근거 선별 출처: "lexical"(KeywordEvidenceSelector) | "cross_encoder"
    # (CrossEncoderEvidenceSelector). 내부 trace/품질 메타일 뿐 외부 계약/후보 순위와 무관.
    rerank_source: str = "lexical"


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
            papers = self._rank_publications(candidate.top_papers, keywords)
            projects = self._rank_projects(candidate.top_projects, keywords)
            patents = self._rank_patents(candidate.top_patents, keywords)

            bundle = RelevantEvidenceBundle(
                expert_id=candidate.expert_id,
                papers=papers,
                projects=projects,
                patents=patents,
            )
            bundles[candidate.expert_id] = bundle

            candidate_count = len(bundle.all_items())
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
            "mode": "keyword_lexical_branch_limits",
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
            if not matched_keywords:
                continue
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
            if not matched_keywords:
                continue
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
            if not matched_keywords:
                continue
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


# 근거(evidence) 종류: (표시 라벨, CandidateCard 소스 속성)
_EVIDENCE_TYPES: tuple[tuple[str, str], ...] = (
    ("paper", "top_papers"),
    ("project", "top_projects"),
    ("patent", "top_patents"),
)


def _sigmoid(value: float) -> float:
    """cross-encoder raw logit → (0,1) 정규화 점수. floor/정렬 모두 정규화 점수 기준."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class CrossEncoderEvidenceSelector:
    """후보 내부 근거(chunk)를 cross-encoder 관련도로 재정렬해 family별 top-N만 추리는 selector.

    WO-0 스텁: 주입된 ``scorer``만 사용하며 실제 모델 로드/DI 와이어링/family cap 실제 적용은
    WO-C에서 채운다. ``scorer`` 부재·예외 시 lexical ``fallback``(KeywordEvidenceSelector)으로 강등.

    HARD 제약(가드):
      - 본 selector는 **근거(grounding) 선별만** 한다 — 후보(연구자) 순위·탈락·생성에 영향 0.
      - 후보 cross-encoder 리랭커(``NTIS_CANDIDATE_RERANKER``, 기본 OFF)와는 **완전 별개**다.
    """

    def __init__(
        self,
        *,
        scorer: object | None,
        fallback: EvidenceSelector,
        top_n_per_type: int = 5,
        relevance_floor: float = 0.30,
        pregate_per_type: int = 20,
        max_pairs_per_request: int = 256,
    ) -> None:
        self.scorer = scorer
        self.fallback = fallback
        self.top_n_per_type = top_n_per_type
        self.relevance_floor = relevance_floor
        self.pregate_per_type = pregate_per_type
        self.max_pairs_per_request = max_pairs_per_request
        self.last_trace: dict[str, object] = {}

    def select(
        self,
        *,
        candidates: list[CandidateCard],
        plan: PlannerOutput,
    ) -> dict[str, RelevantEvidenceBundle]:
        query = self._resolve_query(plan)

        if self.scorer is None:
            return self._lexical_fallback(
                "no_scorer", candidates=candidates, plan=plan, query=query
            )
        return self._rerank(candidates=candidates, plan=plan, query=query)

    # -- query -----------------------------------------------------------
    @staticmethod
    def _resolve_query(plan: PlannerOutput) -> str:
        """semantic_query 우선, 없으면 core_keywords 결합."""
        semantic = (getattr(plan, "semantic_query", "") or "").strip()
        if semantic:
            return semantic
        return " ".join(plan.core_keywords or []).strip()

    # -- fallback --------------------------------------------------------
    def _lexical_fallback(
        self,
        reason: str,
        *,
        candidates: list[CandidateCard],
        plan: PlannerOutput,
        query: str,
    ) -> dict[str, RelevantEvidenceBundle]:
        bundles = self.fallback.select(candidates=candidates, plan=plan)
        fallback_trace = getattr(self.fallback, "last_trace", {}) or {}
        self.last_trace = {
            "mode": "lexical_fallback",
            "fallback_reason": reason,
            "query": query,
            "candidate_evidence_counts": fallback_trace.get(
                "candidate_evidence_counts", []
            ),
        }
        return bundles

    # -- cross-encoder rerank -------------------------------------------
    def _rerank(
        self,
        *,
        candidates: list[CandidateCard],
        plan: PlannerOutput,
        query: str,
    ) -> dict[str, RelevantEvidenceBundle]:
        # 1) dedup → pregate, 그리고 scorer에 보낼 pair 평탄화
        pairs: list[tuple[str, str]] = []
        pair_index: list[tuple[int, str, int, object]] = []  # (cand_i, type, orig_index, item)
        dedup_dropped_by_candidate: list[int] = []

        for candidate_index, candidate in enumerate(candidates):
            dedup_dropped = 0
            for type_label, source_attr in _EVIDENCE_TYPES:
                source = list(getattr(candidate, source_attr, []) or [])
                deduped, dropped = self._dedup(source, type_label)
                dedup_dropped += dropped
                for orig_index, item in deduped[: self.pregate_per_type]:
                    pairs.append((query, self._doc_text(item, type_label)))
                    pair_index.append((candidate_index, type_label, orig_index, item))
            dedup_dropped_by_candidate.append(dedup_dropped)

        # 2) scoring (모델 호출 — 예외 시 lexical fallback)
        try:
            raw_scores = self._score_pairs(pairs)
        except Exception:  # noqa: BLE001 — 어떤 scorer 오류든 안전하게 강등
            return self._lexical_fallback(
                "scorer_error", candidates=candidates, plan=plan, query=query
            )

        # 3) 후보×타입별로 점수 분배
        scored: dict[int, dict[str, list[tuple[int, object, float]]]] = {}
        for (candidate_index, type_label, orig_index, item), raw in zip(pair_index, raw_scores):
            normalized = _sigmoid(float(raw))
            scored.setdefault(candidate_index, {}).setdefault(type_label, []).append(
                (orig_index, item, normalized)
            )

        # 4) floor drop → 정렬 → top-N cap → bundle
        bundles: dict[str, RelevantEvidenceBundle] = {}
        candidate_evidence_counts: list[dict[str, object]] = []

        for candidate_index, candidate in enumerate(candidates):
            floor_dropped = 0
            items_by_type: dict[str, list[RelevantEvidenceItem]] = {}
            for type_label, _ in _EVIDENCE_TYPES:
                entries = scored.get(candidate_index, {}).get(type_label, [])
                kept: list[tuple[int, object, float]] = []
                for orig_index, item, normalized in entries:
                    if normalized < self.relevance_floor:
                        floor_dropped += 1
                        continue
                    kept.append((orig_index, item, normalized))
                kept.sort(
                    key=lambda triple: (
                        -triple[2],
                        -(_parse_year(self._date(triple[1], type_label)) or 0),
                        self._title(triple[1], type_label),
                    )
                )
                items_by_type[type_label] = [
                    self._build_item(orig_index, item, normalized, type_label)
                    for orig_index, item, normalized in kept[: self.top_n_per_type]
                ]

            bundle = RelevantEvidenceBundle(
                expert_id=candidate.expert_id,
                papers=items_by_type["paper"],
                projects=items_by_type["project"],
                patents=items_by_type["patent"],
            )
            bundles[candidate.expert_id] = bundle
            candidate_evidence_counts.append(
                {
                    "expert_id": candidate.expert_id,
                    "papers": len(bundle.papers),
                    "projects": len(bundle.projects),
                    "patents": len(bundle.patents),
                    "total": len(bundle.all_items()),
                    "dedup_dropped": dedup_dropped_by_candidate[candidate_index],
                    "dropped_below_floor": floor_dropped,
                }
            )

        self.last_trace = {
            "mode": "cross_encoder",
            "query": query,
            "scorer": getattr(self.scorer, "model_name", None),
            "candidate_evidence_counts": candidate_evidence_counts,
        }
        return bundles

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """max_pairs_per_request 단위로 배치 호출. 빈 입력이면 scorer 미호출."""
        scores: list[float] = []
        batch = max(1, self.max_pairs_per_request)
        for start in range(0, len(pairs), batch):
            scores.extend(self.scorer.score(pairs[start : start + batch]))  # type: ignore[union-attr]
        return scores

    # -- dedup -----------------------------------------------------------
    def _dedup(
        self, source: list[object], type_label: str
    ) -> tuple[list[tuple[int, object]], int]:
        """동일 (정규화 title, 연도) 중복 제거. (남은 [(orig_index, item)], dropped) 반환."""
        seen: set[tuple[str, int | None]] = set()
        out: list[tuple[int, object]] = []
        dropped = 0
        for orig_index, item in enumerate(source):
            key = (
                _normalize_text(self._title(item, type_label)),
                _parse_year(self._date(item, type_label)),
            )
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            out.append((orig_index, item))
        return out, dropped

    # -- per-type field accessors ---------------------------------------
    @staticmethod
    def _title(item: object, type_label: str) -> str:
        if type_label == "paper":
            return item.publication_title
        if type_label == "project":
            return item.display_title
        return item.intellectual_property_title

    @staticmethod
    def _date(item: object, type_label: str) -> str | None:
        if type_label == "paper":
            return item.publication_year_month
        if type_label == "project":
            return item.project_end_date or item.project_start_date
        return item.registration_date or item.application_date

    @staticmethod
    def _detail(item: object, type_label: str) -> str | None:
        if type_label == "paper":
            return item.journal_name
        if type_label == "project":
            return item.managing_agency or item.performing_organization
        return item.application_registration_type or item.application_country

    @staticmethod
    def _doc_text(item: object, type_label: str) -> str:
        if type_label == "paper":
            parts = [
                item.publication_title,
                item.abstract,
                item.journal_name,
                " ".join(item.korean_keywords),
                " ".join(item.english_keywords),
            ]
        elif type_label == "project":
            parts = [
                item.display_title,
                item.research_objective_summary,
                item.research_content_summary,
                item.managing_agency,
                item.performing_organization,
            ]
        else:
            parts = [
                item.intellectual_property_title,
                item.application_registration_type,
                item.application_country,
            ]
        return " ".join(part for part in parts if part)

    def _build_item(
        self, orig_index: int, item: object, normalized: float, type_label: str
    ) -> RelevantEvidenceItem:
        # NOTE: item_id는 WO-0에서 v1.x 위치 기반 형식을 유지한다(chunk_id 교체는 WO-C).
        return RelevantEvidenceItem(
            item_id=f"{type_label}:{orig_index}",
            type=type_label,
            title=self._title(item, type_label),
            date=self._date(item, type_label),
            detail=self._detail(item, type_label),
            snippet=None,
            matched_keywords=[],
            match_score=normalized,
            rerank_source="cross_encoder",
        )
